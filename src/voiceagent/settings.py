import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# BAAI/bge-base-en-v1.5, run locally (Together has no serverless embedding model).
EMBEDDING_DIM = 768

# Cartesia "Jacqueline": the voice of every call unless the job picks another.
DEFAULT_VOICE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"


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
    # The assessment's reviewer agent challenges rejections: a stronger model than the
    # one that made the call, so it doesn't just agree with itself.
    reviewer_model: str = "deepseek-ai/DeepSeek-V4-Pro-0813"
    # Outcome email. Without SMTP_HOST, mail is only recorded (messages table and log).
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_starttls: bool = True
    # TypeSafe's Jev for rubric scoring, reviewer checks and live question progress.
    # Without it the LLM scorer, LLM reviewer and word-overlap matcher are used.
    typesafe_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/screening"
        ),
        api_key=os.environ.get("SCREENING_API_KEY", ""),
        together_api_key=os.environ.get("TOGETHER_API_KEY", ""),
        smtp_host=os.environ.get("SMTP_HOST", ""),
        smtp_port=int(os.environ.get("SMTP_PORT", "587")),
        smtp_username=os.environ.get("SMTP_USERNAME", ""),
        smtp_password=os.environ.get("SMTP_PASSWORD", ""),
        smtp_from=os.environ.get("SMTP_FROM", ""),
        smtp_starttls=os.environ.get("SMTP_STARTTLS", "true").lower() not in ("0", "false", "no"),
        typesafe_api_key=os.environ.get("TYPESAFE_API_KEY", ""),
    )
