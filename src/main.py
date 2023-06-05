import uvicorn
from app import app
from core.config import APIConfig


if __name__ == "__main__":
    config = APIConfig()
    uvicorn.run(app, host=config.host, port=config.port)