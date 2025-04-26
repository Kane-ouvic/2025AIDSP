# 使用 NVIDIA 的基礎映像，支援 CUDA
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安裝基本依賴
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
RUN pip3 install --no-cache-dir \
    torch==1.10.0 \
    torchvision==0.11.0 \
    opencv-python==4.5.5.64 \
    mediapipe==0.8.9.1 \
    PyQt5==5.15.6 \
    ultralytics==8.0.196 \
    joblib==1.2.0 \
    numpy==1.21.6

# 設定工作目錄
WORKDIR /app

# 複製專案檔案
COPY . /app/

# 設定環境變數
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# 設定容器啟動時的指令
CMD ["python3", "Project2/combine/camera_demo.py", "demo", "--model", "./models/21styles.model", "--style-folder", "./images/custom"]