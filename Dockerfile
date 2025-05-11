# 1) 베이스 이미지 (CPU 전용)
FROM python:3.9-slim

# 2) 작업 디렉토리
WORKDIR /app

# 3) 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) 소스 복사
COPY . .

# 5) 포트 (FastAPI 기본 8000)
EXPOSE 8000

# 6) 컨테이너 실행 커맨드
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]