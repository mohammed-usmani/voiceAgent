import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# BAAI/bge-base-en-v1.5, run locally (Together has no serverless embedding model).
EMBEDDING_DIM = 768


@dataclass(frozen=True)
class Settings:
    database_url: str
    api_key: str
    together_api_key: str
    # Chosen on the criteria agent's own task (10 Together models, 2 job descriptions):
    # DeepSeek-V4-Flash was the fastest full run (31-72 s) at the lowest price, with every
    # requirement grounded; MiniMax-M3 was the most consistent (73-74 s, no errors).
    main_model: str = "deepseek-ai/DeepSeek-V4-Flash-0731"
    backup_model: str = "MiniMaxAI/MiniMax-M3"
    embedding_model: str = "BAAI/bge-base-en-v1.5"


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/screening"
        ),
        api_key=os.environ.get("SCREENING_API_KEY", ""),
        together_api_key=os.environ.get("TOGETHER_API_KEY", ""),
    )
