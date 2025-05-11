from fastapi import FastAPI
import torch
from encoder.jamo_encoder import VECTOR_SIZE
from mask.word_mask import word_mask_window2
from profanity_cnn import ProfanityCNN
from MaskRequest import MaskRequest
from MaskResponse import MaskResponse
import re

app = FastAPI(title="CleanBot API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProfanityCNN(input_dim=VECTOR_SIZE).to(device)
model.load_state_dict(
    torch.load("model/clean_bot_model_v3.pt", map_location=device)
)
model.eval()

def normalize_text(text: str) -> str:
    return re.sub(r'([가-힣])[^가-힣\s]+([가-힣])', r'\1\2', text)

@app.post("/mask", response_model=MaskResponse)
def mask_text(req: MaskRequest):
    raw = req.text
    norm = normalize_text(raw)

    masked = word_mask_window2(raw, model, device)

    return MaskResponse(
        original=raw,
        masked=masked
    )