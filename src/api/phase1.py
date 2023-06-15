from api.request import Request
from api.response import Response
from fastapi import APIRouter, HTTPException
from core.phase1 import Prob1Model, Prob2Model


router = APIRouter()

@router.post("/prob-1/predict", response_model=Response)
async def phase1_prob1(request: Request) -> Response:
    try:
        model = Prob1Model()
        response = model.infer(request=request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/prob-2/predict", response_model=Response)
async def phase1_prob1(request: Request) -> Response:
    try:
        model = Prob2Model()
        response = model.infer(request=request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))