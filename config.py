# Configuration file for Enhanced RAG Chatbot
# Domain-focused for mortgage/fee documents. Clean output and structured "Explain Document" summaries.

# --------------------------- Configuration ---------------------------
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# --------------------------- System dependencies ---------------------------
# Note: Install required packages manually:
# pip install gradio pypdf pdf2image pytesseract pillow nltk rank-bm25 faiss-cpu
# pip install sentence-transformers transformers accelerate scikit-learn
# pip install google-generativeai llama-index llama-index-llms-gemini
# pip install SpeechRecognition pyttsx3 deep-translator langdetect
# pip install pyngrok  # For public tunneling

# Domain heuristics (mortgage focus)
MORTGAGE_KEYWORDS = {
    "fees worksheet","loan estimate","closing disclosure","closing costs","lender fees","origination",
    "underwriting fee","wire transfer fee","administration fee","appraisal fee","credit report",
    "tax service","flood certification","loan amount","interest rate","apr","term","year fixed",
    "borrower","seller","escrow","prepaid","points","mortgage","lender"
}

MORTGAGE_NEGATIVE_KEYWORDS = {"labour and employment act", "employment act", "witness whereof"}

# Performance settings
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_USE_EXPANSION = True
DEFAULT_ENABLE_RERANKING = True
DEFAULT_MAX_TOKENS = 0
DEFAULT_SHOW_DEBUG = True

# OCR settings
DEFAULT_DPI = 300
OCR_CONFIG = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:()[]{}\"' -/$%&*+=<>@#^_|\\~` "

# Chunking settings
TARGET_TOKENS = 300
OVERLAP_SENTENCES = 2
BOUNDARY_DROP = 0.15
TOK_CHARS_RATIO = 0.25

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi'
}

# Export formats
SUPPORTED_EXPORT_FORMATS = ["txt", "pdf", "docx"]

# Error recovery settings
MAX_RETRIES = 3
