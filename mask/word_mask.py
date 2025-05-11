# 단어 단위로 욕설 여부 판별 함수
from sympy.printing.pytorch import torch
from encoder.jamo_encoder import text_to_tensor

def is_offensive_word(word, model, device, threshold=0.7):
    x = text_to_tensor(word, max_len=600)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        # model_v3 사용 시
        logit = model(x_tensor)
        prob = torch.sigmoid(logit).item()
        # prob = model(x_tensor).item()
    return prob > threshold

def word_mask(text, model, device):
    words = text.split()
    masked_words = ["***" if is_offensive_word(w, model, device) else w for w in words]
    masked = " ".join(masked_words)
    return masked

def word_mask_window2(text, model, device):
    words = text.split()
    masked_flags = [False] * len(words)  # 각 단어가 마스킹될지 여부
    masked_words = words.copy()
    if len(words) == 1:
        if is_offensive_word(words[0], model, device):
            return "***"
        else:
            return words[0]

    for i in range(len(words) - 1):
        combined = words[i] + words[i+1]
        if is_offensive_word(combined, model, device):
            if is_offensive_word(words[i], model, device):
                masked_flags[i] = True
                masked_words[i] = "***"
            else:
                masked_flags[i+1] = True
                masked_words[i+1] = "***"
    if is_offensive_word(words[-1], model, device):
        masked_flags[-1] = True
        masked_words[-1] = "***"

    return " ".join(masked_words)