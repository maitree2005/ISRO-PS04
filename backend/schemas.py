from pydantic import BaseModel
from typing import List, Any

class PipelineResponse(BaseModel):
    message: str
    file: str
