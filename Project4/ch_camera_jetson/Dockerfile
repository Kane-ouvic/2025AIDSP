# jetson-containers build script will automatically set this from a pre-built base image.
# See https://github.com/dusty-nv/jetson-containers
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# 設定工作目錄
WORKDIR /app

# 設定 DEBIAN_FRONTEND 為 noninteractive，避免 apt-get 卡在互動式提問
ENV DEBIAN_FRONTEND=noninteractive

# l4t-pytorch 基礎映像檔已包含大部分 GUI 和 Python 相關函式庫，
# 通常不需要手動安裝。如果有缺少，可以取消註解下面的指令。
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libx11-6 \
#     libxcb1 \
#     libxext6 \
#     libxrandr2 \
#     libxi6 \
#     libxrender1 \
#     libfontconfig1 \
#     libsm6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# 基礎映像檔已包含 PyTorch，我們只需要安裝 requirements.txt 中定義的其他套件。

# 複製 requirements.txt 並安裝所有指定的 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 將應用程式程式碼複製到映像中
COPY . .

# 聲明容器預計會監聽的網路連接埠
EXPOSE 7860

# 設定容器啟動時要執行的指令
CMD ["python", "gui_control_1.py"]
