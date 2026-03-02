"""LangGraph state schemas for the translation workflow."""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from src.models.document import (
    ExtractedDocument,
    TranslatedChunk,
    TranslationChunk,
)


class GlossaryEntry(BaseModel):
    """A single entry in the translation glossary."""

    original: str = Field(description="Original term or name")
    translation: str = Field(description="Translated term or name")
    category: str = Field(
        default="term",
        description="Category: 'name', 'place', 'term', 'concept'",
    )
    notes: str = Field(default="", description="Additional context or notes")


class Glossary(BaseModel):
    """Translation glossary and style guide."""

    entries: list[GlossaryEntry] = Field(
        default_factory=list,
        description="List of glossary entries",
    )
    tone: str = Field(
        default="neutral",
        description="Tone of the document (formal, informal, neutral, academic)",
    )
    genre: str = Field(
        default="general",
        description="Genre of the document (fiction, technical, academic, etc.)",
    )
    language_variant: str = Field(
        default="",
        description="Specific language variant (e.g., 'Brazilian Portuguese', 'Mexican Spanish')",
    )
    style_notes: str = Field(
        default="",
        description="Additional style guidance for translation",
    )
    source_language: str = Field(
        default="",
        description="Detected source language of the document",
    )
    forbidden_source_words: list[str] = Field(
        default_factory=list,
        description="Common source-language words that must never appear in the translation",
    )

    def to_prompt(self) -> str:
        """Convert glossary to a prompt-friendly format."""
        lines = [
            f"Document tone: {self.tone}",
            f"Document genre: {self.genre}",
        ]

        if self.language_variant:
            lines.extend([
                "",
                "=== DIALECT / LANGUAGE VARIANT (MANDATORY) ===",
                f"Target variant: {self.language_variant}",
                "You MUST use vocabulary, spelling, grammar, and pronoun forms specific to this variant consistently throughout the ENTIRE translation.",
                "Do NOT mix dialect forms. Every sentence must conform to the same variant.",
            ])

        if self.source_language:
            lines.extend([
                "",
                "=== SOURCE LANGUAGE CONTAMINATION PREVENTION ===",
                f"Source language: {self.source_language}",
                "CRITICAL: The translation must contain ZERO words from the source language.",
                "Do not leave any source-language words untranslated. Every word must be in the target language.",
                "Pay special attention to common false cognates and words that look similar between languages.",
            ])

        if self.forbidden_source_words:
            lines.append(f"Common source words that MUST NOT appear in the output: {', '.join(self.forbidden_source_words)}")

        if self.style_notes:
            lines.extend(["", f"Style notes: {self.style_notes}"])

        if self.entries:
            lines.append("\nGlossary of terms (use these translations consistently):")
            for entry in self.entries:
                line = f"  - {entry.original} → {entry.translation}"
                if entry.notes:
                    line += f" ({entry.notes})"
                lines.append(line)

        return "\n".join(lines)


def merge_translated_chunks(
    existing: list[TranslatedChunk],
    new: list[TranslatedChunk],
) -> list[TranslatedChunk]:
    """Merge new translated chunks with existing ones, preferring newer versions."""
    new_ids = {chunk.id for chunk in new}
    merged = [chunk for chunk in existing if chunk.id not in new_ids]
    merged.extend(new)
    return merged


OutputFormat = Literal["pdf", "docx"]


class TranslationState(TypedDict, total=False):
    """State schema for the LangGraph translation workflow."""

    input_path: str
    target_language: str
    output_format: OutputFormat
    document: ExtractedDocument
    glossary: Glossary
    chunks: list[TranslationChunk]
    translated_chunks: Annotated[list[TranslatedChunk], merge_translated_chunks]
    output_path: str
    error: str | None
