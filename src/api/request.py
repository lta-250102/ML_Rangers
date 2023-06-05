from pydantic import BaseModel


class Phase1Prob1Request(BaseModel):
    id: str
    columns: list[str]
    rows: list[list]