import uvicorn
from src.app import app
from src.core.config import Config


if __name__ == "__main__":
    config = Config('API')
    uvicorn.run(app, host=config.get('host'), port=config.get('port'))