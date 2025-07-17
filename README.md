# Paper Processor

A scientific paper processing system that extracts sections, claims, and analyzes valence from research papers about psychedelics and other drugs. The system uses Neo4j for data storage, OpenAI for claim extraction, and various ML models for valence analysis.

## Features

- **PDF Processing**: Extract text and sections from scientific papers using GROBID, PyMuPDF, or regex fallback
- **Claim Extraction**: Extract scientific claims using OpenAI GPT models
- **Valence Analysis**: Analyze therapeutic vs. abusive effects using multiple approaches:
  - OpenAI GPT-based analysis
  - Zero-shot classification with BART
- **Graph Database**: Store papers, claims, sections, and relationships in Neo4j
- **Statistical Analysis**: Compare different valence analysis methods

## Prerequisites

- Python 3.8+
- Neo4j database
- OpenAI API key
- Docker (for GROBID)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd paper_processor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Configuration

Edit `config.py` to configure:
- Neo4j connection settings
- OpenAI API key
- File paths for data
- Processing limits

## Running GROBID with Docker

GROBID is used for advanced PDF section extraction. To run it with Docker:

1. Pull the GROBID image:
```bash
docker pull lfoppiano/grobid:0.7.3
```

2. Run GROBID container:
```bash
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.3
```

3. Verify GROBID is running:
```bash
curl http://localhost:8070/api/isalive
```

GROBID will be available at `http://localhost:8070` and the system will automatically use it for PDF processing.

## Usage

### Basic Usage

```python
from paper_processor import PaperProcessor

# Initialize processor
processor = PaperProcessor()

# Process papers from CSV file
results = processor.process_csv(
    csv_file="path/to/papers.csv",
    pdf_dir="path/to/pdfs/",
    drug_name="ketamine",
    limit=10  # Optional: limit number of papers
)

# Close connections
processor.close()
```

### Analyze Valence Differences

```python
from analyze_claim_valence import ValenceAnalyzer

analyzer = ValenceAnalyzer()
analyzer.analyze_claim_valences()
analyzer.close()
```

This will generate visualizations and statistics in the `valence_analysis/` directory.

## CSV Format

The system expects a CSV file with the following columns:
- `title`: Paper title
- `authors`: Author names (comma-separated)
- `year`: Publication year
- `journal`: Journal name
- `doi`: DOI identifier
- `abstract`: Paper abstract
- `file_attachments1`: PDF filename (optional)

## Database Schema

The system creates a graph database with the following node types:
- **Paper**: Research papers with metadata
- **Section**: Document sections (abstract, methods, results, etc.)
- **Claim**: Extracted scientific claims
- **Drug**: Drugs being studied
- **Author**: Paper authors
- **Keyword**: Keywords/topics

Relationships include:
- `DISCUSSES`: Paper → Drug
- `HAS_SECTION`: Paper → Section
- `CONTAINS_CLAIM`: Section → Claim
- `WRITTEN_BY`: Paper → Author
- `HAS_KEYWORD`: Paper → Keyword
