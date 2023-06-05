from pydantic import BaseModel


class Phase1Prob1Response(BaseModel):
    id: str
    prediction: list[float]
    drift: int