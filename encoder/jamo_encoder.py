import numpy as np

CHO_LIST = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNG_LIST = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONG_LIST = [''] + ['ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

# 전체 자모 + 영어 + 숫자 + 기호 등 포함한 집합
ALL_JAMO = sorted(set(CHO_LIST + JUNG_LIST + JONG_LIST[1:] +
                      list("abcdefghijklmnopqrstuvwxyz0123456789.,!?~@#$%^&*()[]{}:;\"'<>/-+= ")))
JAMO2IDX = {j: i for i, j in enumerate(ALL_JAMO)}
VECTOR_SIZE = len(JAMO2IDX)

def decompose_char(c):
    code = ord(c)
    if 0xAC00 <= code <= 0xD7A3:
        base = code - 0xAC00
        cho = base // 588
        jung = (base % 588) // 28
        jong = base % 28
        return [CHO_LIST[cho], JUNG_LIST[jung]] + ([JONG_LIST[jong]] if jong != 0 else [])
    else:
        return [c]

def text_to_tensor(text, max_len=600):
    jamo_seq = []
    for c in text:
        jamo_seq.extend(decompose_char(c))
    
    if len(jamo_seq) > max_len:
        jamo_seq = jamo_seq[:max_len]
    else:
        jamo_seq += ['<PAD>'] * (max_len - len(jamo_seq))

    tensor = np.zeros((max_len, VECTOR_SIZE), dtype=np.float32)
    for i, jamo in enumerate(jamo_seq):
        if jamo in JAMO2IDX:
            tensor[i][JAMO2IDX[jamo]] = 1.0
    return tensor
