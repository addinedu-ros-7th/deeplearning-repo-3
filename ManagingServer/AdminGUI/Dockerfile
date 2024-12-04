# 1. 베이스 이미지 설정 (우분투 22.04)
FROM ubuntu:22.04

# 2. 환경 변수 설정 (비대화형 설치용)
ENV DEBIAN_FRONTEND=noninteractive

# 3. 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    && apt-get clean

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. 필요한 파일 복사
COPY requirements.txt .
COPY . .

# 6. Python 패키지 설치
RUN pip install --upgrade pip && pip install -r requirements.txt

# 7. 컨테이너 실행 명령어
CMD ["python3", "AdminGUI.py"]