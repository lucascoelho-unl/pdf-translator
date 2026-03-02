"""LangGraph workflow for PDF translation."""

import logging
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents.glossary_agent import build_glossary
from src.agents.translator_agent import translate_chunks, validate_translations
from src.config import get_settings
from src.models.state import TranslationState
from src.services.chunker import create_translation_chunks
from src.services.extractor import DocumentExtractor
from src.models.state import OutputFormat
from src.services.rebuilder import DocumentRebuilder

logger = logging.getLogger(__name__)


async def extract_document(state: TranslationState) -> dict:
    """
    LangGraph node: Extract document content using DLA.

    Phase 1 of the pipeline - extracts text blocks, images, and layout
    information from the input PDF.
    """
    input_path = state["input_path"]
    logger.info(f"Phase 1: Extracting document from {input_path}")

    extractor = DocumentExtractor()
    document = extractor.extract(input_path)

    pages_with_blocks = set()
    for block in document.blocks:
        pages_with_blocks.add(block.page_number)

    pages_without_text = set(range(1, document.total_pages + 1)) - pages_with_blocks
    if pages_without_text:
        logger.warning(
            f"Pages with NO extracted text blocks: {sorted(pages_without_text)}. "
            f"These pages may contain only images or be blank."
        )

    logger.info(
        f"Extracted {len(document.blocks)} blocks and {len(document.images)} images "
        f"from {document.total_pages} pages "
        f"(text found on {len(pages_with_blocks)}/{document.total_pages} pages)"
    )

    return {"document": document}


async def chunk_document(state: TranslationState) -> dict:
    """
    LangGraph node: Create translation chunks from extracted blocks.

    Prepares text blocks for translation by adding context windows
    for narrative continuity.
    """
    document = state["document"]
    logger.info("Creating translation chunks...")

    chunks = create_translation_chunks(document)

    total_blocks = len(document.blocks)
    skipped = total_blocks - len(chunks)
    logger.info(
        f"Created {len(chunks)} chunks for translation "
        f"({skipped} blocks skipped as headers/footers/page numbers)"
    )

    if len(chunks) == 0:
        logger.error("No translatable chunks were created from the document!")
    elif len(chunks) < total_blocks * 0.5:
        logger.warning(
            f"Only {len(chunks)}/{total_blocks} blocks will be translated. "
            f"Check if the extractor is capturing all content."
        )

    return {"chunks": chunks}


async def rebuild_document_node(state: TranslationState) -> dict:
    """
    LangGraph node: Rebuild document with translated content.

    Phase 4 of the pipeline - generates a new PDF or DOCX from translated
    content.
    """
    settings = get_settings()
    document = state["document"]
    translated_chunks = state["translated_chunks"]
    target_language = state["target_language"]
    output_format: OutputFormat = state.get("output_format", "pdf")

    input_path = Path(state["input_path"])
    extension = ".docx" if output_format == "docx" else ".pdf"
    output_filename = f"{input_path.stem}_{target_language}{extension}"
    output_path = settings.output_dir / output_filename

    total_blocks = len(document.blocks)
    translated_count = len(translated_chunks)
    failed_count = sum(
        1 for c in translated_chunks if c.translated_text.startswith("[TRANSLATION ERROR")
    )
    logger.info(
        f"Phase 4: Rebuilding {output_format.upper()} to {output_path} "
        f"({translated_count} translated chunks, {failed_count} failed, "
        f"{total_blocks} total blocks)"
    )

    rebuilder = DocumentRebuilder()
    final_path = rebuilder.rebuild(
        document=document,
        translated_chunks=translated_chunks,
        output_path=output_path,
        target_language=target_language,
        output_format=output_format,
    )

    logger.info(f"Translation complete: {final_path}")

    return {"output_path": str(final_path)}


def should_continue(state: TranslationState) -> str:
    """Determine if the workflow should continue or end."""
    if state.get("error"):
        return "error"
    return "continue"


def build_translation_graph() -> StateGraph:
    """
    Build the complete LangGraph translation workflow.

    The workflow consists of four phases:
    1. Extract: Document Layout Analysis
    2. Glossary: Build translation context
    3. Translate: Parallel chunk translation
    4. Rebuild: Generate output PDF

    Returns:
        Compiled LangGraph StateGraph
    """
    workflow = StateGraph(TranslationState)

    workflow.add_node("extract_document", extract_document)
    workflow.add_node("build_glossary", build_glossary)
    workflow.add_node("chunk_document", chunk_document)
    workflow.add_node("translate_chunks", translate_chunks)
    workflow.add_node("validate_translations", validate_translations)
    workflow.add_node("rebuild_pdf", rebuild_document_node)

    workflow.set_entry_point("extract_document")

    workflow.add_edge("extract_document", "build_glossary")
    workflow.add_edge("build_glossary", "chunk_document")
    workflow.add_edge("chunk_document", "translate_chunks")
    workflow.add_edge("translate_chunks", "validate_translations")
    workflow.add_edge("validate_translations", "rebuild_pdf")
    workflow.add_edge("rebuild_pdf", END)

    return workflow.compile()


async def run_translation(
    input_path: str | Path,
    target_language: str,
    output_format: OutputFormat = "pdf",
) -> str:
    """
    Run the complete translation pipeline.

    Args:
        input_path: Path to the input PDF
        target_language: Target language code (e.g., 'pt-BR', 'es', 'fr')
        output_format: Output format ('pdf' or 'docx')

    Returns:
        Path to the generated translated document
    """
    graph = build_translation_graph()

    initial_state: TranslationState = {
        "input_path": str(input_path),
        "target_language": target_language,
        "output_format": output_format,
    }

    result = await graph.ainvoke(initial_state)

    return result["output_path"]
