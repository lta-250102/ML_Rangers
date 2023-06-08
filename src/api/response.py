from pydantic import BaseModel


class Response(BaseModel):
    id: str
    prediction: list[float]
    drift: int