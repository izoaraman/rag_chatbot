# LANGUAGE_MODEL = "google/flan-t5-base"
# LANGUAGE_MODEL = "google/flan-t5-xxl"
# LANGUAGE_MODEL = "bigscience/bloom-560m"
# LANGUAGE_MODEL = "EleutherAI/gpt-neo-125M"
# EMBEDDING_MODEL = 'multi-qa-mpnet-base-dot-v1'
# LANGUAGE_MODEL = "cerebras/Cerebras-GPT-2.7B"
# LANGUAGE_MODEL = "EleutherAI/gpt-neo-1.3B"
# LANGUAGE_MODEL = "gpt2-medium"
# LANGUAGE_MODEL = "google/electra-large-discriminator"
# global_settings.py

LANGUAGE_MODEL = "gpt2-medium"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

MAX_NEW_TOKENS = 100  # Increase this for generating more text
MAX_LENGTH = 100 # Control the total length of the generated response
TEMPERATURE = 0.5  # Adjust the randomness of the output, lower temperature mean more focused and factual response
TOP_K = 5  # Top-K sampling for text generation
TOP_P = 0.7  # Nucleus sampling (Top-P) for text generation

# Paths and configurations
LOG_FILE = "session_data/user_actions.log"
SESSION_FILE = "session_data/user_session_state.yaml"
CACHE_FILE = "cache/pipeline_cache.json"
CONVERSATION_FILE = "cache/chat_history.json"
QUIZ_FILE = "cache/quiz.csv"
SLIDES_FILE = "cache/slides.json"
STORAGE_PATH = "ingestion_storage/"
INDEX_STORAGE = "index_storage"
SUPPORTED_FILE_TYPES = ['pdf']

QUIZ_SIZE = 5
ITEMS_ON_SLIDE = 4
