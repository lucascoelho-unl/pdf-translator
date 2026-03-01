"""Intelligent chunking service for translation preparation."""

import logging

from src.models.document import ExtractedDocument, TextBlock, TranslationChunk

logger = logging.getLogger(__name__)


def create_translation_chunks(
    document: ExtractedDocument,
    previous_translations: dict[str, str] | None = None,
    context_window: int = 2,
) -> list[TranslationChunk]:
    """
    Create translation chunks from extracted document blocks.

    Each chunk includes context from previously translated blocks to maintain
    narrative continuity during translation.

    Args:
        document: The extracted document with text blocks
        previous_translations: Map of block IDs to their translations (for context)
        context_window: Number of previous blocks to include as context

    Returns:
        List of translation chunks ready for the translation agent
    """
    previous_translations = previous_translations or {}

    sorted_blocks = sorted(
        document.blocks,
        key=lambda b: (b.page_number, b.reading_order),
    )

    chunks: list[TranslationChunk] = []
    recent_translations: list[str] = []

    for block in sorted_blocks:
        if not block.text.strip():
            continue

        if _should_skip_block(block):
            continue

        context = ""
        if recent_translations:
            context_blocks = recent_translations[-context_window:]
            context = "\n\n".join(context_blocks)

        chunk = TranslationChunk(
            id=block.id,
            text=block.text,
            block_type=block.block_type,
            previous_context=context,
        )
        chunks.append(chunk)

        if block.id in previous_translations:
            recent_translations.append(previous_translations[block.id])

    logger.info(f"Created {len(chunks)} translation chunks from {len(sorted_blocks)} blocks")
    return chunks


def _should_skip_block(block: TextBlock) -> bool:
    """Determine if a block should be skipped for translation."""
    if block.block_type in ("header", "footer"):
        if len(block.text) < 50:
            return True

    if block.text.strip().isdigit():
        return True

    return False


def group_chunks_by_page(chunks: list[TranslationChunk]) -> dict[int, list[TranslationChunk]]:
    """Group translation chunks by their page number."""
    pass


def estimate_token_count(text: str) -> int:
    """Rough estimate of token count for a text string."""
    return len(text) // 4
