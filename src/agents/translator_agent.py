"""Translation agent with parallel processing and rate limiting."""

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm.asyncio import tqdm_asyncio

from src.config import get_settings
from src.models.document import TranslatedChunk, TranslationChunk
from src.models.state import Glossary, TranslationState

logger = logging.getLogger(__name__)

TRANSLATION_SYSTEM_PROMPT = """You are an expert publication-quality translator. Your translations must be indistinguishable from text originally written by a native speaker of the target language.

CORE PRINCIPLES:
1. Translate naturally and idiomatically — NEVER translate literally word-by-word
2. Maintain the original meaning, nuance, and emotional impact
3. Preserve the tone, register, and style of the original
4. Use the provided glossary strictly for consistent terminology
5. Maintain narrative flow using the provided context

MANDATORY RULES:
- DIALECT CONSISTENCY: Use ONLY the dialect/variant specified in the glossary. Every pronoun, verb conjugation, and vocabulary choice must conform to the SAME dialect throughout. NEVER mix regional forms (e.g., never mix "vosotros" with "ustedes").
- ZERO SOURCE LANGUAGE CONTAMINATION: Your output must contain ABSOLUTELY NO words from the source language. Every single word must be in the target language. Double-check that no source-language words were accidentally left in the text.
- CLINICAL/TECHNICAL TERMS: Use the OFFICIAL terminology from the glossary. For psychological terms (e.g., Schema Therapy), use the academically established translations — never literal translations.
- BIBLICAL CITATIONS: If the text contains Bible verses, use the OFFICIAL published Bible version in the target language (as specified in the glossary, e.g., NVI or Reina-Valera 1960 for Spanish). Do NOT translate Bible verses literally from the source — reproduce the exact text from the official version.
- Preserve formatting (paragraphs, lists, etc.)
- Do not add explanations, notes, or translator comments — output ONLY the translation
- If the text contains code, URLs, or technical identifiers, keep them unchanged
- Do not omit or skip any part of the source text — translate EVERYTHING"""


def get_translation_prompt(
    text: str,
    target_language: str,
    glossary: Glossary,
    previous_context: str = "",
    block_type: str = "paragraph",
) -> str:
    """Generate the translation prompt with context."""
    prompt_parts = [
        f"Translate the following {block_type} to {target_language}.",
        "",
        glossary.to_prompt(),
    ]

    if previous_context:
        prompt_parts.extend([
            "",
            "PREVIOUS TRANSLATED TEXT (for context continuity):",
            previous_context,
        ])

    prompt_parts.extend([
        "",
        "TEXT TO TRANSLATE:",
        text,
        "",
        "TRANSLATION:",
    ])

    return "\n".join(prompt_parts)


MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2


async def translate_single_chunk(
    chunk: TranslationChunk,
    target_language: str,
    glossary: Glossary,
    llm: ChatGoogleGenerativeAI,
    semaphore: asyncio.Semaphore,
) -> TranslatedChunk:
    """
    Translate a single chunk with rate limiting and automatic retry.

    Args:
        chunk: The chunk to translate
        target_language: Target language for translation
        glossary: Glossary for consistent terminology
        llm: The LLM client
        semaphore: Semaphore for rate limiting

    Returns:
        Translated chunk
    """
    async with semaphore:
        prompt = get_translation_prompt(
            text=chunk.text,
            target_language=target_language,
            glossary=glossary,
            previous_context=chunk.previous_context,
            block_type=chunk.block_type,
        )

        messages = [
            SystemMessage(content=TRANSLATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await llm.ainvoke(messages)
                content = response.content
                if isinstance(content, list):
                    translated_text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    ).strip()
                else:
                    translated_text = str(content).strip()

                return TranslatedChunk(
                    id=chunk.id,
                    original_text=chunk.text,
                    translated_text=translated_text,
                    block_type=chunk.block_type,
                )
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    backoff_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        f"Chunk {chunk.id[:8]}... failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. "
                        f"Retrying in {backoff_time}s..."
                    )
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(
                        f"Chunk {chunk.id[:8]}... failed after {MAX_RETRIES + 1} attempts: {e}"
                    )

        return TranslatedChunk(
            id=chunk.id,
            original_text=chunk.text,
            translated_text=f"[TRANSLATION ERROR: {chunk.text}]",
            block_type=chunk.block_type,
        )


async def translate_chunks(state: TranslationState) -> dict:
    """
    LangGraph node that translates all chunks in parallel with rate limiting.

    Uses asyncio.Semaphore to limit concurrent API calls and prevent
    rate limit errors.

    Args:
        state: Current translation state with chunks and glossary

    Returns:
        Updated state with translated chunks
    """
    settings = get_settings()
    chunks = state["chunks"]
    glossary = state["glossary"]
    target_language = state["target_language"]

    logger.info(f"Translating {len(chunks)} chunks to {target_language}...")

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )

    semaphore = asyncio.Semaphore(settings.max_concurrent_translations)

    tasks = [
        translate_single_chunk(
            chunk=chunk,
            target_language=target_language,
            glossary=glossary,
            llm=llm,
            semaphore=semaphore,
        )
        for chunk in chunks
    ]

    translated_chunks = await tqdm_asyncio.gather(
        *tasks,
        desc="Translating",
        unit="chunk",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    success_count = sum(
        1 for c in translated_chunks if not c.translated_text.startswith("[TRANSLATION ERROR")
    )
    logger.info(f"Successfully translated {success_count}/{len(chunks)} chunks")

    return {"translated_chunks": list(translated_chunks)}


def _check_source_contamination(
    text: str,
    forbidden_words: list[str],
) -> list[str]:
    """Check translated text for source-language word contamination."""
    found = []
    text_lower = text.lower()
    words_in_text = set(text_lower.split())
    for word in forbidden_words:
        if word.lower() in words_in_text:
            found.append(word)
    return found


async def validate_translations(state: TranslationState) -> dict:
    """
    LangGraph node that validates translation quality.

    Checks for:
    - Missing translations
    - Suspiciously short or long translations
    - Source-language contamination
    - Failed translation chunks (retries them)

    Args:
        state: Current translation state with translated chunks

    Returns:
        Updated state with retried chunks if needed
    """
    settings = get_settings()
    translated_chunks = list(state["translated_chunks"])
    chunks = state["chunks"]
    glossary = state["glossary"]
    target_language = state["target_language"]

    original_ids = {c.id for c in chunks}
    translated_ids = {c.id for c in translated_chunks}

    missing = original_ids - translated_ids
    if missing:
        logger.warning(f"Missing translations for {len(missing)} chunks")

    forbidden_words = glossary.forbidden_source_words if glossary else []

    issues = []
    failed_chunk_ids = set()
    contaminated_chunk_ids = set()

    for chunk in translated_chunks:
        if chunk.translated_text.startswith("[TRANSLATION ERROR"):
            issues.append(f"Chunk {chunk.id[:8]}: Translation failed")
            failed_chunk_ids.add(chunk.id)
            continue

        original_len = len(chunk.original_text)
        translated_len = len(chunk.translated_text)

        if translated_len < original_len * 0.3:
            issues.append(f"Chunk {chunk.id[:8]}: Translation suspiciously short ({translated_len} vs {original_len} chars)")
        elif translated_len > original_len * 3:
            issues.append(f"Chunk {chunk.id[:8]}: Translation suspiciously long ({translated_len} vs {original_len} chars)")

        if forbidden_words:
            contamination = _check_source_contamination(chunk.translated_text, forbidden_words)
            if contamination:
                issues.append(
                    f"Chunk {chunk.id[:8]}: Source language contamination detected: {contamination}"
                )
                contaminated_chunk_ids.add(chunk.id)

    if issues:
        logger.warning(f"Translation validation found {len(issues)} issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    chunks_to_retry_ids = failed_chunk_ids | contaminated_chunk_ids
    if chunks_to_retry_ids:
        logger.info(f"Retrying {len(chunks_to_retry_ids)} problematic chunks...")

        chunks_to_retry = [c for c in chunks if c.id in chunks_to_retry_ids]

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )
        semaphore = asyncio.Semaphore(settings.max_concurrent_translations)

        retry_tasks = [
            translate_single_chunk(
                chunk=chunk,
                target_language=target_language,
                glossary=glossary,
                llm=llm,
                semaphore=semaphore,
            )
            for chunk in chunks_to_retry
        ]

        retried_results = await asyncio.gather(*retry_tasks)

        retried_map = {c.id: c for c in retried_results}
        final_chunks = []
        for chunk in translated_chunks:
            if chunk.id in retried_map:
                retried = retried_map[chunk.id]
                if not retried.translated_text.startswith("[TRANSLATION ERROR"):
                    final_chunks.append(retried)
                    logger.info(f"Chunk {chunk.id[:8]}: Retry succeeded")
                else:
                    final_chunks.append(chunk)
                    logger.warning(f"Chunk {chunk.id[:8]}: Retry also failed")
            else:
                final_chunks.append(chunk)

        success_rate = sum(
            1 for c in final_chunks if not c.translated_text.startswith("[TRANSLATION ERROR")
        ) / max(len(final_chunks), 1) * 100
        logger.info(f"Final translation success rate: {success_rate:.1f}%")

        return {"translated_chunks": final_chunks}

    success_rate = sum(
        1 for c in translated_chunks if not c.translated_text.startswith("[TRANSLATION ERROR")
    ) / max(len(translated_chunks), 1) * 100
    logger.info(f"Translation validation complete. Success rate: {success_rate:.1f}%")

    return {}
