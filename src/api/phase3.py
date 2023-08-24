import logging
from api.request import Request
from api.response import Response
from fastapi import APIRouter, HTTPException
from core.phase3 import Prob1Model, Prob2Model
from core.save_test import save_test, save_test_executor


router = APIRouter()
logger = logging.getLogger("ml_ranger_logger")

@router.post("/prob-1/predict", response_model=Response)
async def phase1_prob1(request: Request) -> Response:
    try:
        model = Prob1Model()
        response = model.infer(request=request)
        save_test_executor.submit(save_test, request, response, 'phase-3', 'prob-1')
        response = Response(
            id=request.id,
            predictions=[0] * len(request.rows),
            drift=0
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/prob-2/predict", response_model=Response)
async def phase1_prob2(request: Request) -> Response:
    try:
        model = Prob2Model()
        response = model.infer(request=request)
        save_test_executor.submit(save_test, request, response, 'phase-3', 'prob-2')
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))