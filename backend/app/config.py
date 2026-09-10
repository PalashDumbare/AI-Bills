from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://palash@localhost:5432/ai_bills"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
