import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

# Model config
MODEL = "qwen/qwen3-32b"
MAX_TOKENS = 4096

# Database
DB_PATH = os.path.join(os.path.dirname(__file__), "state", "study.db")
