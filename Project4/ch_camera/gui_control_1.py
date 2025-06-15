import gradio as gr
import torch
import time
import logging
from PIL import Image, ImageOps
import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import ImageSegmenter
from DeepCache.sd.pipeline_stable_diffusion import StableDiffusionPipeline as DeepCacheStableDiffusionPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全域模型變數 ---
pipe_inpaint = None
selfie_segmenter = None
face_detector = None
pipe_controlnet = None
openpose_detector = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(model_path, progress=gr.Progress(track_tqdm=True)):
    """從指定路徑載入模型"""
    progress(0, desc="正在載入模型...")
    try:
        pipe = DeepCacheStableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:0")
        logging.info(f"從 '{model_path}' 載入模型成功。")
        progress(1, desc="模型載入成功！")
        return pipe
    except Exception as e:
        logging.error(f"模型載入失敗: {e}")
        raise gr.Error(f"模型載入失敗，請檢查路徑或網路連線: {e}")

def load_face_detector():
    """Lazy-load FaceDetection model"""
    global face_detector
    if face_detector is None:
        logging.info("正在載入 FaceDetection 模型...")
        try:
            face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
            logging.info("FaceDetection 模型載入成功。")
        except Exception as e:
            logging.error(f"FaceDetection 模型載入失敗: {e}")
            raise gr.Error(f"無法載入人臉偵測模型: {e}")

def set_random_seed(seed):
    """設定隨機種子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- 圖片生成函式 (DeepCache) ---
def generate_with_deepcache(pipe, prompt, height, width, seed, cache_interval, pow_val, center_val, progress=gr.Progress(track_tqdm=True)):
    """根據輸入參數使用 DeepCache 生成圖片"""
    if pipe is None:
        raise gr.Error("請先載入 DeepCache 模型。")

    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    set_random_seed(int(seed))
    
    logging.info("開始使用 DeepCache 生成圖片...")
    start_time = time.time()
    
    # 使用 DeepCache 生成圖片
    output = pipe(
        prompt,
        height=height,
        width=width,
        cache_interval=int(cache_interval),
        cache_layer_id=0,
        cache_block_id=0,
        uniform=False,
        pow=pow_val,
        center=int(center_val),
        output_type='pil',
        return_dict=True
    ).images[0]
    
    use_time = time.time() - start_time
    logging.info(f"DeepCache 生成耗時: {use_time:.2f} 秒")
    
    info_text = f"Seed: {int(seed)}\n耗時: {use_time:.2f} 秒"
    
    return output, info_text

# --- 圖片生成函式 (Inpainting) ---
def generate_with_inpainting(snapshot, prompt, mask_choice, seed, progress=gr.Progress(track_tqdm=True)):
    """使用 Inpainting 模型及 Mediapipe 遮罩來生成圖片"""
    global pipe_inpaint, selfie_segmenter, face_detector

    if snapshot is None:
        raise gr.Error("請先從攝影機拍攝一張照片。")

    # 1. 載入模型 (若尚未載入)
    progress(0, desc="正在載入模型...")
    if pipe_inpaint is None:
        try:
            pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", 
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(device)
        except Exception as e:
            logging.error(f"Inpainting 模型載入失敗: {e}")
            raise gr.Error(f"模型載入失敗，請檢查網路連線: {e}")
    
    if selfie_segmenter is None:
        try:
            # 使用 model_selection=0 適用於一般近照，1 適用於風景照中的人物
            selfie_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        except Exception as e:
            logging.error(f"Mediapipe 載入失敗: {e}")
            raise gr.Error(f"無法載入影像分割模型: {e}")

    progress(0.1, desc="準備生成...")
    # 2. 設定隨機種子
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(int(seed))

    # 3. 使用 Mediapipe 進行影像分割以取得遮罩
    progress(0.2, desc="正在進行影像分割...")
    
    # Mediapipe 需要 RGB 格式的 numpy array
    rgb_image = snapshot

    print(mask_choice)
    
    if mask_choice == "遮罩全身(保留頭)":
        progress(0.25, desc="SelfieSegmentation 取人物...")
        # 1. 整人二元遮罩
        seg_result = selfie_segmenter.process(snapshot)
        mask_np = (seg_result.segmentation_mask > 0.5).astype(np.uint8) * 255

        # 2. 找人臉框
        progress(0.35, desc="FaceDetection 擷取頭部...")
        load_face_detector()
        bgr = cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR)
        fd_result = face_detector.process(bgr)
        if not fd_result.detections:
            raise gr.Error("未偵測到人臉，請調整角度/光線再試。")

        # 3. 把最大人臉 bbox 當「頭」，多放大一些確保髮型
        h, w, _ = snapshot.shape
        det = max(fd_result.detections, key=lambda d: d.location_data.relative_bounding_box.width)
        rb = det.location_data.relative_bounding_box
        # 轉絕對座標並放大
        x0 = max(int((rb.xmin - 0.15*rb.width)  * w), 0)
        y0 = max(int((rb.ymin - 0.15*rb.height) * h), 0)
        x1 = min(int((rb.xmin + rb.width*1.15) * w), w)
        y1 = min(int((rb.ymin + rb.height*1.3) * h), h)

        # 4. 將頭部區域清零 → 只留身體
        mask_np[y0:y1, x0:x1] = 0

        # 5. (可選) 膨脹身體遮罩，避免邊緣殘留
        mask_np = cv2.dilate(mask_np, np.ones((10,10), np.uint8), iterations=1)
        mask_image = Image.fromarray(mask_np)

    else:
        progress(0.2, desc="正在進行影像分割...")
        results = selfie_segmenter.process(rgb_image)
        
        # 將 Mediapipe 的輸出 (0.0-1.0) 轉換為二元遮罩 (0 或 255)
        mask_condition = results.segmentation_mask > 0.5
        mask_np = np.where(mask_condition, 255, 0).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)

        if mask_choice == "遮罩人物":
            mask_image = mask_pil
        else:  # "遮罩背景"
            mask_image = ImageOps.invert(mask_pil.convert('L'))

    # 準備輸入 Inpainting 管線的圖片
    image_pil = Image.fromarray(snapshot)
    image = image_pil.resize((512, 512))
    # 確保遮罩圖大小正確，並儲存一份用於顯示
    display_mask = mask_image.resize((512, 512))

    # 4. 執行 Inpainting
    progress(0.5, desc="正在執行 Inpainting...")
    start_time = time.time()
    
    output = pipe_inpaint(
        prompt=prompt,
        image=image,
        mask_image=display_mask, # 使用調整好大小的遮罩
        generator=generator,
        output_type='pil'
    ).images[0]

    use_time = time.time() - start_time
    logging.info(f"Inpainting 生成耗時: {use_time:.2f} 秒")
    info_text = f"Seed: {int(seed)}\n耗時: {use_time:.2f} 秒"

    return output, info_text, display_mask

# --- 骨架控制生成函式 (ControlNet) ---
def generate_with_openpose(person_image, prompt, seed, progress=gr.Progress(track_tqdm=True)):
    """使用 ControlNet (OpenPose) 實現骨架控制生成"""
    global pipe_controlnet, openpose_detector

    if person_image is None:
        raise gr.Error("請先從攝影機拍攝一張人像照片。")

    # 1. 載入模型 (若尚未載入)
    progress(0, desc="正在載入 ControlNet 模型...")
    if openpose_detector is None:
        try:
            # For better performance, you can move this model loading part to a separate function or on app start.
            openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            logging.info("OpenPose Detector 載入成功。")
        except Exception as e:
            logging.error(f"OpenPose Detector 載入失敗: {e}")
            raise gr.Error(f"無法載入 OpenPose 模型: {e}")

    if pipe_controlnet is None:
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16
            )
            pipe_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(device)
            logging.info("ControlNet Pipeline 載入成功。")
        except Exception as e:
            logging.error(f"ControlNet Pipeline 載入失敗: {e}")
            raise gr.Error(f"無法載入 SD+ControlNet 模型: {e}")

    progress(0.2, desc="準備生成...")
    # 2. 設定隨機種子
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(int(seed))

    # 3. 使用 OpenPose 提取人體姿勢
    progress(0.3, desc="正在提取人體姿勢...")
    person_image_pil = Image.fromarray(person_image)
    # The resolution can be adjusted based on input image quality and desired performance
    pose_image = openpose_detector(person_image_pil, detect_resolution=384, image_resolution=512)
    
    # 4. 執行 ControlNet Pipeline
    progress(0.5, desc="正在生成試衣圖片...")
    start_time = time.time()
    
    # Using a smaller number of inference steps for faster generation, can be increased for higher quality.
    output = pipe_controlnet(
        prompt,
        image=pose_image,
        generator=generator,
        num_inference_steps=20, 
        output_type='pil'
    ).images[0]

    use_time = time.time() - start_time
    logging.info(f"ControlNet 生成耗時: {use_time:.2f} 秒")
    info_text = f"Seed: {int(seed)}\n耗時: {use_time:.2f} 秒"

    return output, pose_image, info_text

# --- Gradio 介面 ---
with gr.Blocks() as demo:
    pipe_state = gr.State(None)

    gr.Markdown("# Stable Diffusion 智慧圖片生成器")
    gr.Markdown("支援 DeepCache 加速的文字生成圖片，以及使用攝影機畫面進行的智慧修補。")
    
    with gr.Tabs():
        with gr.TabItem("攝影機智慧修補 (Webcam Inpainting)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 輸入影像")
                    webcam_input = gr.Image(sources=["webcam", "upload"], label="輸入畫面", type="numpy")
                    gr.Markdown("可使用攝影機拍照或直接上傳圖片。")

                    gr.Markdown("### 2. 設定並生成")
                    prompt_inpaint_input = gr.Textbox(label="提示詞 (Prompt)", value="a high-quality, detailed photograph of a person in a futuristic city")
                    mask_choice_input = gr.Radio(["遮罩背景", "遮罩人物", "遮罩全身(保留頭)"], label="遮罩選項 (Masking Option)", value="遮罩背景")
                    inpaint_seed_input = gr.Number(label="種子 (Seed)", value=42, precision=0, info="-1 代表隨機")
                    generate_inpaint_btn = gr.Button("生成圖片", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 3. 查看成果")
                    image_output_inpaint = gr.Image(label="生成結果")
                    mask_output_inpaint = gr.Image(label="產生的遮罩")
                    info_output_inpaint = gr.Textbox(label="生成資訊")

        with gr.TabItem("骨架控制生成 (Pose Control)"):
            gr.Markdown("### 使用 ControlNet + OpenPose 進行骨架控制生成")
            gr.Markdown("1. 拍攝或上傳一張包含完整人體的照片。\n2. 在提示詞中詳細描述您想生成的場景與人物。\n3. 點擊生成！")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 1. 輸入")
                    pose_person_image_input = gr.Image(sources=["webcam", "upload"], label="拍攝/上傳人像", type="numpy", height=400)
                    prompt_openpose_input = gr.Textbox(label="提示詞 (Prompt)", value="a high-quality photo of a woman wearing a red dress, full body, standing in a showroom")
                    openpose_seed_input = gr.Number(label="種子 (Seed)", value=42, precision=0, info="-1 代表隨機")
                    generate_openpose_btn = gr.Button("生成圖片", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 2. 結果")
                    image_output_openpose = gr.Image(label="生成結果", height=400)
                    pose_image_output = gr.Image(label="偵測到的人體姿勢 (ControlNet Input)")
                    info_output_openpose = gr.Textbox(label="生成資訊")

        with gr.TabItem("文字生成圖片 (DeepCache)"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("1. 模型設定", open=True):
                        model_path_input = gr.Textbox(
                            label="模型路徑 (Model Path)", 
                            value="runwayml/stable-diffusion-v1-5",
                            info="請輸入 Hugging Face 模型 ID 或本地端已下載模型的資料夾路徑。"
                        )
                        load_model_btn = gr.Button("載入模型", variant="secondary")

                    gr.Markdown("### 2. 生成設定")
                    prompt_deepcache_input = gr.Textbox(label="提示詞 (Prompt)", value="a photo of an astronaut on a moon")
                    
                    with gr.Accordion("進階選項", open=False):
                        height_input = gr.Slider(256, 1024, value=512, step=64, label="高度 (Height)")
                        width_input = gr.Slider(256, 1024, value=512, step=64, label="寬度 (Width)")
                        seed_input = gr.Number(label="種子 (Seed)", value=42, precision=0, info="-1 代表隨機")
                        
                        gr.Markdown("#### DeepCache 參數")
                        cache_interval_input = gr.Slider(1, 10, value=5, step=1, label="快取間隔 (Cache Interval)")
                        pow_val_input = gr.Slider(1.0, 2.0, value=1.4, step=0.1, label="冪次 (Power)")
                        center_val_input = gr.Slider(1, 30, value=15, step=1, label="中心 (Center)")

                    generate_deepcache_btn = gr.Button("使用 DeepCache 生成", variant="primary")
                    
                with gr.Column(scale=1):
                    image_output_deepcache = gr.Image(label="生成結果")
                    info_output_deepcache = gr.Textbox(label="生成資訊")

    # --- 事件綁定 ---
    # DeepCache Tab
    load_model_btn.click(
        fn=load_model,
        inputs=[model_path_input],
        outputs=[pipe_state]
    )
            
    generate_deepcache_btn.click(
        fn=generate_with_deepcache,
        inputs=[
            pipe_state,
            prompt_deepcache_input, height_input, width_input, seed_input, 
            cache_interval_input, pow_val_input, center_val_input
        ],
        outputs=[image_output_deepcache, info_output_deepcache]
    )

    # Inpainting Tab
    generate_inpaint_btn.click(
        fn=generate_with_inpainting,
        inputs=[webcam_input, prompt_inpaint_input, mask_choice_input, inpaint_seed_input],
        outputs=[image_output_inpaint, info_output_inpaint, mask_output_inpaint]
    )

    # Pose Control Tab
    generate_openpose_btn.click(
        fn=generate_with_openpose,
        inputs=[pose_person_image_input, prompt_openpose_input, openpose_seed_input],
        outputs=[image_output_openpose, pose_image_output, info_output_openpose]
    )

if __name__ == "__main__":
    demo.launch()
