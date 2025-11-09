from pydantic import BaseModel
from typing import List


class GuardrailsData(BaseModel):
    name: str
    shape: List[int]
    data: List
    datatype: str


class GuardrailsRequest(BaseModel):
    inputs: List[GuardrailsData]


class GuardrailsResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[GuardrailsData]
