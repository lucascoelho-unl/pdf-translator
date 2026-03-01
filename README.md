# PDF Translator

AI-powered PDF translation system that preserves document layout using LangGraph and Google Gemini.

## Features

- **Layout-Aware Extraction**: Uses Document Layout Analysis (DLA) to extract text with structural metadata
- **Context-Aware Translation**: Builds a glossary and style guide to ensure consistent translations
- **Parallel Processing**: Translates chunks concurrently with rate limiting
- **Layout Preservation**: Reconstructs PDFs using HTML/CSS intermediate format

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Phase 1        │     │  Phase 2        │     │  Phase 3        │     │  Phase 4        │
│  Extraction     │────▶│  Glossary       │────▶│  Translation    │────▶│  Rebuild        │
│  (Docling DLA)  │     │  (Gemini)       │     │  (Parallel)     │     │  (WeasyPrint)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Installation

1. Clone the repository:

```bash
git clone <repo-url>
cd pdf-translator
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Usage

### Basic Usage

```bash
python -m src.main data/your-document.pdf --language pt-BR
```

### Options

```
positional arguments:
  input_pdf             Path to the input PDF file

options:
  -l, --language LANG   Target language code (e.g., pt-BR, es, fr, de)
  -o, --output PATH     Output PDF path (optional, auto-generated if not specified)
  -v, --verbose         Enable verbose logging
```

### Examples

```bash
# Translate to Brazilian Portuguese
python -m src.main data/book.pdf --language pt-BR

# Translate to Spanish with verbose logging
python -m src.main data/manual.pdf --language es --verbose

# Translate to French
python -m src.main data/article.pdf --language fr
```

## Project Structure

```
pdf-translator/
├── src/
│   ├── main.py              # CLI entry point
│   ├── config.py            # Settings and environment
│   ├── models/
│   │   ├── document.py      # Document, Block, Image models
│   │   └── state.py         # LangGraph state schemas
│   ├── agents/
│   │   ├── graph.py         # LangGraph workflow
│   │   ├── glossary_agent.py    # Context extraction
│   │   └── translator_agent.py  # Translation nodes
│   ├── services/
│   │   ├── extractor.py     # PDF extraction with DLA
│   │   ├── chunker.py       # Intelligent chunking
│   │   └── rebuilder.py     # PDF reconstruction
│   └── templates/
│       └── document.html    # Jinja2 template for PDF
├── data/                    # Input PDFs
├── output/                  # Translated PDFs
├── .env.example
├── requirements.txt
└── README.md
```

## Configuration

| Variable                      | Description               | Default            |
| ----------------------------- | ------------------------- | ------------------ |
| `GOOGLE_API_KEY`              | Gemini API key (required) | -                  |
| `TARGET_LANGUAGE`             | Default target language   | `pt-BR`            |
| `LOG_LEVEL`                   | Logging level             | `INFO`             |
| `MAX_CONCURRENT_TRANSLATIONS` | Parallel API calls        | `5`                |
| `GEMINI_MODEL`                | Gemini model to use       | `gemini-2.0-flash` |

## How It Works

### Phase 1: Document Layout Analysis

Uses Docling to extract text blocks with bounding boxes, element types (headings, paragraphs, captions), and reading order.

### Phase 2: Glossary Building

An LLM analyzes sample text to identify:

- Character and place names
- Technical terminology
- Document tone and genre
- Style guidelines

### Phase 3: Parallel Translation

Text chunks are translated concurrently with:

- Glossary injection for consistency
- Sliding window context for narrative flow
- Rate limiting to prevent API throttling

### Phase 4: PDF Reconstruction

Generates a new PDF by:

1. Building semantic HTML from translated blocks
2. Applying CSS for layout styling
3. Rendering to PDF with WeasyPrint

This approach allows natural text reflow when translations expand or contract.

## License

MIT
