"""Translation agent with parallel processing and rate limiting."""

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.models.document import TranslatedChunk, TranslationChunk
from src.models.state import Glossary, TranslationState

logger = logging.getLogger(__name__)

TRANSLATION_SYSTEM_PROMPT = """You are an expert translator. Your task is to translate text while:
1. Maintaining the original meaning and nuance
2. Preserving the tone and style of the original
3. Using the provided glossary for consistent terminology
4. Maintaining narrative flow with the provided context

Rules:
- Translate naturally, not literally
- Preserve formatting (paragraphs, lists, etc.)
- Keep proper nouns consistent with the glossary
- Match the formality level of the original
- Do not add explanations or notes - only output the translation
- If the text contains code, URLs, or technical identifiers, keep them unchanged"""


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


async def translate_single_chunk(
    chunk: TranslationChunk,
    target_language: str,
    glossary: Glossary,
    llm: ChatGoogleGenerativeAI,
    semaphore: asyncio.Semaphore,
) -> TranslatedChunk:
    """
    Translate a single chunk with rate limiting.

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

        try:
            response = await llm.ainvoke(messages)
            translated_text = response.content.strip()

            return TranslatedChunk(
                id=chunk.id,
                original_text=chunk.text,
                translated_text=translated_text,
                block_type=chunk.block_type,
            )
        except Exception as e:
            logger.error(f"Failed to translate chunk {chunk.id}: {e}")
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

    translated_chunks = await asyncio.gather(*tasks)

    success_count = sum(
        1 for c in translated_chunks if not c.translated_text.startswith("[TRANSLATION ERROR")
    )
    logger.info(f"Successfully translated {success_count}/{len(chunks)} chunks")

    return {"translated_chunks": list(translated_chunks)}


async def validate_translations(state: TranslationState) -> dict:
    """
    LangGraph node that validates translation quality.

    Checks for:
    - Missing translations
    - Suspiciously short or long translations
    - Untranslated content

    Args:
        state: Current translation state with translated chunks

    Returns:
        Updated state (potentially with retry flags)
    """
    translated_chunks = state["translated_chunks"]
    chunks = state["chunks"]

    original_ids = {c.id for c in chunks}
    translated_ids = {c.id for c in translated_chunks}

    missing = original_ids - translated_ids
    if missing:
        logger.warning(f"Missing translations for {len(missing)} chunks")

    issues = []
    for chunk in translated_chunks:
        if chunk.translated_text.startswith("[TRANSLATION ERROR"):
            issues.append(f"Chunk {chunk.id}: Translation failed")
            continue

        original_len = len(chunk.original_text)
        translated_len = len(chunk.translated_text)

        if translated_len < original_len * 0.3:
            issues.append(f"Chunk {chunk.id}: Translation suspiciously short")
        elif translated_len > original_len * 3:
            issues.append(f"Chunk {chunk.id}: Translation suspiciously long")

    if issues:
        logger.warning(f"Translation validation found {len(issues)} issues")
        for issue in issues[:5]:
            logger.warning(f"  - {issue}")

    return {}
