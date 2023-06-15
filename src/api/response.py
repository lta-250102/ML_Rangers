from pydantic import BaseModel


class Response(BaseModel):
    id: str
    predictions: list[float]
    drift: int