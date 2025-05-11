from pydantic import BaseModel

class MaskResponse(BaseModel):
    original: str
    masked: str