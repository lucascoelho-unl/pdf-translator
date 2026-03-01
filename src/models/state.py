"""LangGraph state schemas for the translation workflow."""

from typing import Annotated, TypedDict

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
    style_notes: str = Field(
        default="",
        description="Additional style guidance for translation",
    )

    def to_prompt(self) -> str:
        """Convert glossary to a prompt-friendly format."""
        lines = [
            f"Document tone: {self.tone}",
            f"Document genre: {self.genre}",
        ]

        if self.style_notes:
            lines.append(f"Style notes: {self.style_notes}")

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
    """Merge new translated chunks with existing ones."""
    existing_ids = {chunk.id for chunk in existing}
    merged = list(existing)
    for chunk in new:
        if chunk.id not in existing_ids:
            merged.append(chunk)
    return merged


class TranslationState(TypedDict, total=False):
    """State schema for the LangGraph translation workflow."""

    input_path: str
    target_language: str
    document: ExtractedDocument
    glossary: Glossary
    chunks: list[TranslationChunk]
    translated_chunks: Annotated[list[TranslatedChunk], merge_translated_chunks]
    output_path: str
    error: str | None
