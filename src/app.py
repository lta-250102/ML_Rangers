from fastapi import FastAPI
from core.logger import init_logger
from api.phase1 import router as phase1_router
from api.phase2 import router as phase2_router
from api.phase3 import router as phase3_router
from core.phase1 import load_model as load_model_phase1
from core.phase2 import load_model as load_model_phase2
from core.phase3 import load_model as load_model_phase3


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, we are ML Rangers!"}

# app.include_router(phase1_router, prefix='/phase-1')
# app.include_router(phase2_router, prefix='/phase-2')
app.include_router(phase3_router, prefix='/phase-3')

@app.on_event("startup")
def init_system_logger():
    init_logger()


load_model_phase3()