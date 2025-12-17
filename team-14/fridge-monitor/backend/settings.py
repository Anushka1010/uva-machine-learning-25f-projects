from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Fridge Monitor API"
    DB_URL: str = "sqlite:///./fridge.db"
    UPLOAD_DIR: str = "./uploads"
    CORS_ORIGINS: str = "http://localhost:3000"

settings = Settings()
