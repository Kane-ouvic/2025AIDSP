# 使用 NVIDIA CUDA 11.7.1 作為基礎映像，以支援 PyTorch 的 CUDA 功能
FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# 設定工作目錄
WORKDIR /app

# 設定 DEBIAN_FRONTEND 為 noninteractive，避免 apt-get 卡在互動式提問
ENV DEBIAN_FRONTEND=noninteractive

# 安裝 Python 和必要的系統函式庫 (給 OpenCV 和 PyQt5)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    libxext6 \
    libxrandr2 \
    libxi6 \
    libxrender1 \
    libfontconfig1 \
    libsm6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 建立 python -> python3.9 的符號連結，並覆蓋現有的連結
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# 為了獲得更好的快取和可維護性，建議將 Python 套件列表放在 requirements.txt 檔案中。
# 此處為求方便，直接寫在 Dockerfile 內。

# 首先安裝 PyTorch 相關套件，指定對應 CUDA 版本的來源
RUN pip install --no-cache-dir torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# 複製 requirements.txt 並安裝所有指定的 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 將應用程式程式碼複製到映像中
COPY . .

# 聲明容器預計會監聽的網路連接埠
EXPOSE 7860

# 設定容器啟動時要執行的指令
CMD ["python", "gui_control_1.py"]
