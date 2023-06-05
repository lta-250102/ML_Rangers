from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.request import Phase1Prob1Request
from api.response import Phase1Prob1Response
from core.phase1 import Prob1Model


router = APIRouter()

@router.post("/1/predict", response_model=Phase1Prob1Response)
def phase1_prob1(request: Phase1Prob1Request) -> Phase1Prob1Response:
    try:
        model = Prob1Model()
        response = model.infer(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.put("/1/train", response_model=None)
def phase1_prob1(background_tasks: BackgroundTasks):
    try:
        model = Prob1Model()
        background_tasks.add_task(model.train)
        return {'message': 'training started in background'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))