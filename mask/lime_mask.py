from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
from encoder.jamo_encoder import text_to_tensor

def lime_predict(texts, model, device):
    model.eval()
    outputs = []
    for text in texts:
        x = text_to_tensor(text, max_len=600)
        x_tensor = torch.tensor(np.array([x]), dtype=torch.float32).to(device)
        with torch.no_grad():
            logit = model(x_tensor)
            prob = torch.sigmoid(logit).item()
        outputs.append([1 - prob, prob])  # [clean, abusive]
    return np.array(outputs)


def lime_mask(text, model, device, threshold=0.2, top_k=10):
    model.eval()

    # 1. 전체 문장 욕설 예측
    x = text_to_tensor(text, max_len=600)
    x_tensor = torch.tensor(np.expand_dims(x, axis=0), dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = model(x_tensor)
        pred_prob = torch.sigmoid(logit).item()

    if pred_prob < 0.4:
        return text  # 욕설 가능성 낮으면 원문 그대로 반환

    # 2. LIME 해석기 생성
    explainer = LimeTextExplainer(class_names=["clean", "abusive"])
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda x: lime_predict(x, model, device),
        num_features=top_k,
        labels=[1]
    )

    # 3. 중요 단어 추출
    important_words = [
        word for word, weight in explanation.as_list(label=1)
        if weight >= threshold
    ]

    # 4. 마스킹 처리
    masked = text
    for word in important_words:
        masked = masked.replace(word, "***")

    return masked