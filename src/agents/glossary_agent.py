"""Glossary extraction agent for building translation context."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.models.state import Glossary, GlossaryEntry, TranslationState

logger = logging.getLogger(__name__)

GLOSSARY_SYSTEM_PROMPT = """You are an expert literary and technical translator assistant specialized in producing publication-quality translations. Your task is to analyze a document sample and extract key information that will ensure translation consistency and professional quality.

Analyze the provided text and identify:

1. **Source language detection**: Identify the source language of the document precisely (e.g., "Brazilian Portuguese", "European Portuguese", "English").

2. **Character names, place names, and proper nouns** that should be translated or transliterated consistently.

3. **Technical/clinical terminology**: Identify ALL domain-specific terms (psychological, theological, medical, legal, etc.) and provide their OFFICIAL translations in the target language. For psychological terms (e.g., Schema Therapy / Terapia de Esquemas), use the officially accepted terminology in the target language's academic literature — do NOT translate literally.

4. **Biblical and religious citations**: If the text contains Bible verses or religious references, identify them. Biblical citations MUST use the official published versions in the target language (e.g., for Spanish: NVI - Nueva Versión Internacional, or Reina-Valera 1960; for Portuguese: NVI or ARA). NEVER translate Bible verses literally from the source — always reference the official version.

5. **The overall tone** (formal, informal, neutral, academic, literary).

6. **The genre** (fiction, non-fiction, technical, academic, legal, self-help, theological, etc.).

7. **Language variant**: Based on the target language code, determine the exact dialect. This is CRITICAL for consistency:
   - For "es": use neutral Latin American Spanish (ustedes, NO vosotros)
   - For "es-ES": use Castilian/Peninsular Spanish (vosotros)
   - For "es-MX": use Mexican Spanish
   - For "pt-BR": use Brazilian Portuguese (você)
   - For "pt-PT": use European Portuguese (tu)
   The entire translation MUST use ONE consistent dialect. Mixing dialects is unacceptable.

8. **Source language interference prevention**: List common words from the source language that a translator might accidentally leave untranslated. These are "false friends" or words that a non-native speaker might forget to translate.

9. **Style notes** for maintaining consistency.

Output your analysis as a JSON object with this exact structure:
{
    "entries": [
        {
            "original": "original term",
            "translation": "official/correct translation in target language",
            "category": "name|place|term|concept|clinical|biblical",
            "notes": "context, e.g. 'Schema Therapy official term' or 'NVI Bible version'"
        }
    ],
    "source_language": "Detected source language (e.g., 'Brazilian Portuguese', 'English')",
    "tone": "formal|informal|neutral|academic|literary",
    "genre": "fiction|non-fiction|technical|academic|legal|self-help|theological|general",
    "language_variant": "Exact language variant to use consistently (e.g., 'Latin American Spanish (neutral)', 'Brazilian Portuguese')",
    "style_notes": "Detailed guidance including: dialect rules (pronoun forms, verb conjugations), clinical terminology standards, biblical citation versions to use, and any other consistency requirements",
    "forbidden_source_words": ["list", "of", "common", "source", "language", "words", "that", "must", "never", "appear", "in", "translation"]
}

Only output valid JSON, no additional text or markdown formatting."""


LANGUAGE_VARIANT_HINTS = {
    "pt-BR": "Brazilian Portuguese (use 'você', Brazilian vocabulary and spelling)",
    "pt-PT": "European Portuguese (use 'tu', Portuguese vocabulary and spelling)",
    "pt": "Brazilian Portuguese (default to Brazilian variant)",
    "es": "Neutral Latin American Spanish (avoid region-specific slang)",
    "es-MX": "Mexican Spanish",
    "es-AR": "Argentine Spanish (use 'vos')",
    "es-ES": "Castilian Spanish (use 'vosotros')",
    "en": "American English",
    "en-US": "American English",
    "en-GB": "British English",
    "fr": "Standard French (France)",
    "fr-CA": "Canadian French",
    "zh-CN": "Simplified Chinese (Mainland China)",
    "zh-TW": "Traditional Chinese (Taiwan)",
}


def get_glossary_prompt(text: str, target_language: str) -> str:
    """Generate the glossary extraction prompt."""
    variant_hint = LANGUAGE_VARIANT_HINTS.get(
        target_language,
        f"Standard {target_language} variant"
    )
    
    return f"""Analyze the following text sample and create a comprehensive translation glossary for translating to {target_language}.

TARGET LANGUAGE VARIANT: {variant_hint}

CRITICAL REQUIREMENTS:
1. DIALECT CONSISTENCY: The ENTIRE translation must use {variant_hint}. If the target is Latin American Spanish, use "ustedes" (NEVER "vosotros"), and conjugate ALL verbs accordingly. Do not mix peninsular and Latin American forms under any circumstances.

2. SOURCE LANGUAGE DETECTION: Identify the source language precisely. Then list common words from that source language that could accidentally leak into the translation (e.g., Portuguese words like "ainda", "então", "mas", "também", "porém", "contudo", "onde", "aqui" that might be left untranslated in a Spanish translation).

3. CLINICAL/TECHNICAL TERMINOLOGY: If the text contains psychological, therapeutic, or clinical terminology (e.g., Schema Therapy, cognitive distortions, attachment styles), provide the OFFICIAL translations used in the target language's academic and clinical literature. Do NOT translate these literally — use the established terms from published works in the target language.

4. BIBLICAL CITATIONS: If the text contains Bible verses or references, note the official Bible version to use in the target language. For Spanish: use NVI (Nueva Versión Internacional) or Reina-Valera 1960. For Portuguese: use NVI or ARA. Biblical text must NEVER be translated literally from the source — the translator must look up the official version.

5. STYLE NOTES: Include detailed guidance on regional vocabulary, pronoun usage, verb conjugation patterns, and any other consistency requirements.

TEXT SAMPLE:
{text}

Output only valid JSON following the specified structure. Be thorough — the quality of this glossary directly determines the publication quality of the translation."""


async def build_glossary(state: TranslationState) -> dict:
    """
    LangGraph node that builds a glossary from the document.

    Scans representative samples from the document to identify:
    - Character and place names
    - Technical terms and concepts
    - Document tone and genre
    - Style guidance

    Args:
        state: Current translation state with extracted document

    Returns:
        Updated state with glossary
    """
    settings = get_settings()
    document = state["document"]
    target_language = state["target_language"]

    logger.info("Building glossary and style guide...")

    sample_text = document.get_text_sample(num_blocks=50)

    if not sample_text.strip():
        logger.warning("No text content found for glossary extraction")
        return {
            "glossary": Glossary(
                entries=[],
                tone="neutral",
                genre="general",
                style_notes="",
            )
        }

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )

    messages = [
        SystemMessage(content=GLOSSARY_SYSTEM_PROMPT),
        HumanMessage(content=get_glossary_prompt(sample_text, target_language)),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            response_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            response_text = str(content)

        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        glossary_data = json.loads(response_text)

        entries = [
            GlossaryEntry(
                original=entry.get("original", ""),
                translation=entry.get("translation", ""),
                category=entry.get("category", "term"),
                notes=entry.get("notes", ""),
            )
            for entry in glossary_data.get("entries", [])
            if entry.get("original")
        ]

        glossary = Glossary(
            entries=entries,
            tone=glossary_data.get("tone", "neutral"),
            genre=glossary_data.get("genre", "general"),
            language_variant=glossary_data.get("language_variant", ""),
            style_notes=glossary_data.get("style_notes", ""),
            source_language=glossary_data.get("source_language", ""),
            forbidden_source_words=glossary_data.get("forbidden_source_words", []),
        )

        logger.info(
            f"Built glossary with {len(entries)} entries, "
            f"tone: {glossary.tone}, genre: {glossary.genre}, "
            f"variant: {glossary.language_variant or 'not specified'}, "
            f"source: {glossary.source_language or 'not detected'}, "
            f"forbidden words: {len(glossary.forbidden_source_words)}"
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse glossary JSON: {e}")
        glossary = Glossary(
            entries=[],
            tone="neutral",
            genre="general",
            style_notes="",
        )
    except Exception as e:
        logger.error(f"Glossary extraction failed: {e}")
        glossary = Glossary(
            entries=[],
            tone="neutral",
            genre="general",
            style_notes="",
        )

    return {"glossary": glossary}
