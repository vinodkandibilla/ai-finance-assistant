"""Package-relative dataset paths, independent of the working directory."""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent
FINANCE_PDF_PATH = DATA_ROOT / "corpora/finance_qa/Financial_Education_Handbook_integrated.pdf"
TAX_CORPUS_PATH = DATA_ROOT / "corpora/tax_education/irs_tax_2026_rag_corpus_expanded.jsonl"
PORTFOLIO_PRICE_SNAPSHOT_PATH = DATA_ROOT / "portfolio/price_snapshot_2026-10-09.json"
