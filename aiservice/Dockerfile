FROM python:3.11

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (중요) OpenCV 런타임 의존성 설치: libGL 등
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# YOLO 설정 디렉토리 생성 및 권한 부여
# 다음의 경고창 해결
# WARNING ⚠️ user config directory '/root/.config/Ultralytics' is not writeable, defaulting to '/tmp' or CWD. Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path.
RUN mkdir -p /root/.config/Ultralytics && chmod -R 777 /root/.config/Ultralytics

# 어플리케이션 복사
COPY . .

# 8001 포트 노출
EXPOSE 8001

# Flask 서버 실행 (Flask 애플리케이션을 실행) : 개발환경
# CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8001"]

# Gunicorn을 사용하여 Flask 앱 실행 (로그 하나도 안남게)
# CMD ["gunicorn", "--bind", "0.0.0.0:8001", "app:app"]

# Gunicorn을 사용하여 Flask 앱 실행 (INFO 로그만 남게. 실제 배포시 사용할 거)
# CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "--timeout", "120", "app:app"]

# Gunicorn을 사용하여 Flask 앱 실행 (모든 로깅 남게. 디버깅용. 배포시에는 위에거 사용)
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "debug", "--timeout", "300", "app:app"]