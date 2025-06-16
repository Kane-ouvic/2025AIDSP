############################################################
#   Stable-Diffusion / Mediapipe / ControlNet on Jetson    #
############################################################
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
WORKDIR /app

# 讓後續 RUN 使用 Bash，便於字串處理
SHELL ["/bin/bash", "-eu", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

# ── A. 基本 X11/GPU 庫 + APT 版 OpenCV 4.5.4 ────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    libx11-6 libxcb1 libxext6 libxrandr2 \
    libxi6 libxrender1 libfontconfig1 libsm6 \
    python3-opencv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── B. 固定 pip / packaging，避免 tzdata 解析 bug ──────────
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip==23.3.1 packaging==23.2

# ── C. 安裝主要依賴（不含 mediapipe / controlnet_aux）──────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── D. 安裝 Mediapipe & ControlNet-Aux，但不解析依賴 ───────
RUN pip install --no-cache-dir --no-deps \
    mediapipe==0.10.9 \
    controlnet_aux==0.0.10

# ── E. 給 cv2/gapi 打補丁 (忽略缺失符號) ────────────────────
RUN python3 - <<'PY'
import cv2, pathlib, re, textwrap
# 判斷 gapi/__init__.py 真正路徑
so_path = pathlib.Path(cv2.__file__).resolve()
gapi_init = (so_path.parent / "cv2" / "gapi" / "__init__.py"
             if so_path.suffix == ".so"
             else so_path.parent / "gapi" / "__init__.py")

txt = gapi_init.read_text(encoding="utf-8")
pat = r"cv\.gapi\.wip\.GStreamerPipeline\s*=\s*cv\.gapi_wip_gst_GStreamerPipeline"
if re.search(pat, txt):
    patched = re.sub(pat,
        textwrap.dedent("""\
        try:
            cv.gapi.wip.GStreamerPipeline = cv.gapi_wip_gst_GStreamerPipeline
        except AttributeError:
            pass"""), txt, count=1)
    gapi_init.write_text(patched, encoding="utf-8")
    print(f"[patch] {gapi_init}")
else:
    print("[patch] no change needed")
PY

# ── F. 雙保險：若有 wheel 版 OpenCV 仍殘留就卸掉 ─────────
RUN python3 - <<'PY'
import pkg_resources, subprocess, sys, re
for dist in list(pkg_resources.working_set):
    if re.match(r"opencv(-python|_python|_contrib_python)", dist.project_name, re.I):
        print("[remove wheel]", dist.project_name, dist.version)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", dist.project_name])
PY

# ── G. 版本列印（可刪） ─────────────────────────────────────
RUN python3 - <<'PY'
import cv2, mediapipe, controlnet_aux, pip, importlib.metadata as im, textwrap
print(textwrap.dedent(f"""
--- Runtime Versions ---
cv2             : {cv2.__version__}  ({cv2.__file__})
mediapipe       : {mediapipe.__version__}
controlnet_aux  : {controlnet_aux.__version__}
pip             : {pip.__version__}
packaging       : {im.version('packaging')}
-------------------------
"""))
PY

# ── H. 拷貝專案程式碼並設定入口 ─────────────────────────────
COPY . .
EXPOSE 7860
CMD ["python3", "gui_control_1.py"]
