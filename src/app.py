from fastapi import FastAPI
from api.phase1 import router as phase1_router


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, we are ML Rangers!"}

app.include_router(phase1_router, prefix='/phase-1')