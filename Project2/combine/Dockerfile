# 使用 NVIDIA 的 L4T PyTorch Base Image (有 CUDA 支援，適合 Jetson Xavier)
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 移除有問題的套件庫並更新
RUN rm -f /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件 (不碰 torch，保留 l4t-pytorch 自帶的 CUDA 版 torch)
RUN pip3 install --no-cache-dir \
    ultralytics==8.0.196 --no-deps \
    joblib==1.2.0 \
    numpy==1.21.6

# (可選) 安裝 OpenCV-Python Headless，如果真的需要 pip 版本
# RUN pip3 install opencv-python-headless

# 設定工作目錄
WORKDIR /app

# 複製專案檔案
COPY . /app/

# 設定環境變數 (顯示用)
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# 容器啟動時的指令
CMD ["python3", "Project2/combine/camera_demo.py", "demo", "--model", "./models/21styles.model", "--style-folder", "./images/custom"]
