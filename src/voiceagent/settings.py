import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# gemini-embedding-001 truncated to 768 dimensions: plenty for short criteria texts,
# and a quarter of the full 3072 in storage and index size.
EMBEDDING_DIM = 768


@dataclass(frozen=True)
class Settings:
    database_url: str
    api_key: str
    gemini_api_key: str
    groq_api_key: str
    main_model: str = "gemini-2.5-flash"
    backup_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "gemini-embedding-001"


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/screening"
        ),
        api_key=os.environ.get("SCREENING_API_KEY", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
    )
