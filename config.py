import os

NEOJ4_URI = "bolt://localhost:7687"
NEOJ4_USER = "neo4j"
NEOJ4_PASSWORD = "password"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV_DIR = os.path.join(PROJECT_ROOT, "data", "csv")
DEFAULT_PDF_DIR = {
    "ketamine": os.path.join(PROJECT_ROOT, "papers-dataset", "Ketamine PDFs"),
    "psilocybin": os.path.join(PROJECT_ROOT, "papers-dataset", "Psilocybin PDFs", "Psilocybin Included PDFs")
}

MAX_CONTENT_LENGTH = 100000
MAX_SECTION_LENGTH = 5000

OPENAI_MODEL = "gpt-4o-mini"
MAX_CLAIMS_PER_SECTION = 3

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = "INFO" 