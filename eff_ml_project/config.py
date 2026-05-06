import os
import re
from pathlib import Path

API_KEY  = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL    = os.getenv("LLM_MODEL")

BATCH_SIZE = 128
SEQ_LEN = 300
N_WORKERS = 16
PCA_COMPONENTS = 50
DATA_PATH = Path(__file__).parent.parent / "data"
FINEWEB_PATH = DATA_PATH / "fineweb10B"
METRICS_PATH = DATA_PATH / "metrics"
DEVICE = "cuda:0"
LAYER = 12
METRIC_COLUMNS = ['random_seed_0',
       'random_seed_1', 'random_seed_2', 'random_seed_3', 'random_seed_4',
       'random_seed_5', 'random_seed_6', 'random_seed_7', 'random_seed_8',
       'random_seed_9', 'schatten_norm', 'effective_rank', 'MLE', 'CorrInt',
       'TwoNN']
NON_RANDOM_METRICS = [m for m in METRIC_COLUMNS if not m.startswith("random_seed_")]

N_EXTREMES = 1000
MAX_CONCURRENT = 8
PARSE_MAX_RETRIES = 5
OUTPUT_CSV = Path(__file__).parent / "pattern_labels.csv"
PATTERNS = {
    "Narrative Prose": (
        "Coherent continuous text with logical progression, contextual continuity, "
        "and structured transitions between ideas. Common in essays, articles, "
        "books, and high-quality authored content."
    ),

    "Named-Entity Flood": (
        "Text overloaded with names of people, organizations, brands, locations, "
        "dates, and other entities, often reducing readability and semantic clarity."
    ),

    "SEO Spam": (
        "Content optimized primarily for search engine ranking rather than human "
        "readability, typically containing keyword stuffing and artificial phrasing."
    ),

    "OCR Noise": (
        "Artifacts produced by optical character recognition, including corrupted "
        "characters, malformed words, missing spaces, and structurally degraded text."
    ),

    "Semantic Incoherence": (
        "Sentences may remain grammatically valid while lacking logical or semantic "
        "coherence across the text."
    ),

    "Word Salad": (
        "Extremely incoherent text in which words and phrases fail to form meaningful "
        "semantic structure."
    ),

    "Metadata Dump": (
        "Large blocks of technical or system-generated information such as logs, "
        "JSON fragments, identifiers, timestamps, and metadata fields."
    ),

    "Mixed-Format": (
        "A document containing multiple heterogeneous formats simultaneously, such as "
        "prose, code, HTML, markdown, tables, or logs."
    ),

    "Repetitive Templates": (
        "Text dominated by repeated structures, sentence patterns, or templated "
        "formulations."
    ),

    "List-like Structure": (
        "Text organized primarily as lists, enumerations, bullet points, or fragmented "
        "entries rather than continuous prose."
    ),
}

def _to_field(name: str) -> str:
    """'Named-Entity Flood' -> 'named_entity_flood' (valid Python identifier)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

PATTERN_NAMES  = list(PATTERNS.keys())
PATTERN_FIELDS = [_to_field(n) for n in PATTERN_NAMES]
FIELD_TO_NAME  = dict(zip(PATTERN_FIELDS, PATTERN_NAMES))

SYSTEM_PROMPT = (
    "You are a text classifier. For each pattern below output true if the text "
    "clearly exhibits that pattern, false otherwise.\n\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in PATTERNS.items())
)