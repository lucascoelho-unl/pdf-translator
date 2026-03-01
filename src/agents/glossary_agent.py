"""Glossary extraction agent for building translation context."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.models.state import Glossary, GlossaryEntry, TranslationState

logger = logging.getLogger(__name__)

GLOSSARY_SYSTEM_PROMPT = """You are an expert literary and technical translator assistant. Your task is to analyze a document sample and extract key information that will ensure translation consistency.

Analyze the provided text and identify:
1. Character names, place names, and proper nouns that should be translated or transliterated consistently
2. Technical terms, concepts, or domain-specific vocabulary
3. The overall tone (formal, informal, neutral, academic, literary)
4. The genre (fiction, non-fiction, technical, academic, legal, etc.)
5. Any style notes that would help maintain consistency

Output your analysis as a JSON object with this exact structure:
{
    "entries": [
        {
            "original": "original term",
            "translation": "suggested translation",
            "category": "name|place|term|concept",
            "notes": "optional context"
        }
    ],
    "tone": "formal|informal|neutral|academic|literary",
    "genre": "fiction|non-fiction|technical|academic|legal|general",
    "style_notes": "Additional guidance for the translator"
}

Only output valid JSON, no additional text or markdown formatting."""


def get_glossary_prompt(text: str, target_language: str) -> str:
    """Generate the glossary extraction prompt."""
    return f"""Analyze the following text sample and create a translation glossary for translating to {target_language}.

TEXT SAMPLE:
{text}

Remember to output only valid JSON following the specified structure."""


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

    sample_text = document.get_text_sample(num_blocks=30)

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
            style_notes=glossary_data.get("style_notes", ""),
        )

        logger.info(
            f"Built glossary with {len(entries)} entries, "
            f"tone: {glossary.tone}, genre: {glossary.genre}"
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
