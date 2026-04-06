from pydantic import BaseModel

class Transaction(BaseModel):
    Time: float
    Amount: float