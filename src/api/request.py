from pydantic import BaseModel


class Request(BaseModel):
    id: str
    columns: list[str]
    rows: list[list]