from pydantic import BaseModel

# 1) 요청 바디 스키마
class MaskRequest(BaseModel):
    text: str