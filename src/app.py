from fastapi import FastAPI
from api.phase1 import router as phase1_router
from core.phase1 import load_model as load_model_phase1


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, we are ML Rangers!"}

app.include_router(phase1_router, prefix='/phase-1')

load_model_phase1()