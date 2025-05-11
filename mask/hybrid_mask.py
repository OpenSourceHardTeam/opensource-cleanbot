from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
from encoder.jamo_encoder import text_to_tensor

# class_names: 0 = 비욕설, 1 = 욕설
class_names = ['clean', 'abusive']

def custom_tokenizer(text):
    return text.split()  # 공백 기준만 사용해서 특수문자 포함 단어를 유지

# LIME용 예측 함수: 리스트[str] → 확률 반환
def lime_predict(texts, model, device):
    results = []
    for t in texts:
        x = text_to_tensor(t, max_len=600)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_tensor).item()
        results.append([1 - pred, pred])
    return np.array(results)


# 단어 단위로 욕설 여부 판별 함수
def is_offensive_word(word, model, device, threshold=0.2):
    x = text_to_tensor(word, max_len=600)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(x_tensor).item()
    return prob > threshold


def hybrid_mask_text(text, model, device, threshold=0.2, lime_weight=0.05):
    # 어절 단위 분리
    words = text.split()

    # 전체 문장에 대해 LIME 실행
    explainer = LimeTextExplainer(class_names=["clean", "abusive"])

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda texts: lime_predict(texts, model, device),
        num_features=10,
        labels=[1]
    )

    # LIME이 기여도가 높다고 판단한 단어 목록
    lime_tokens = [w for w, weight in explanation.as_list(label=1) if weight > lime_weight]

    # 최종 마스킹: 어절 예측 or LIME 기여 둘 중 하나라도 해당되면 마스킹
    masked_words = [
        "***" if is_offensive_word(w, model, device, threshold=threshold) or w in lime_tokens else w
        for w in words
    ]
    return " ".join(masked_words)

