import openpifpaf
import torch
import argparse
import copy
import logging
import torch.multiprocessing as mp
import csv
import os
import matplotlib.pyplot as plt
import cv2
import base64
import time
import numpy as np
import re
import sys
import pandas as pd
from scipy.signal import savgol_filter, lfilter
import math
import io
import PIL
from enum import IntEnum, unique
from typing import List
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import torch.nn as nn


# Parameters and configuration
DEFAULT_CONSEC_FRAMES = 36
MIN_THRESH = 0.5
HIST_THRESH = 0.2

HEAD_THRESHOLD = 1e-5
EMA_FRAMES = DEFAULT_CONSEC_FRAMES * 3
EMA_BETA = 1 / (EMA_FRAMES + 1)
FEATURE_SCALAR = {"ratio_bbox": 1, "gf": 1, "angle_vertical": 1, "re": 1, "ratio_derivative": 1, "log_angle": 1}
FEATURE_LIST = ["ratio_bbox", "log_angle", "re", "ratio_derivative", "gf"]
FRAME_FEATURES = 2

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# === Core algorithms ===
def get_source(args):
    # 這裡僅保留 webcam 即時辨識，忽略影片檔案等其他來源
    cam = cv2.VideoCapture(0)
    ret, img = cam.read()
    if not ret:
        logging.error("無法讀取攝影機影像")
    logging.debug('Image shape: %s', img.shape)
    return cam, None

def resize(img, resize_val, resolution):
    if resize_val is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in resize_val.split('x')]
    width_height = (int(width * resolution // 16) * 16,
                    int(height * resolution // 16) * 16)
    return width, height, width_height

def extract_keypoints_parallel(queue, args, self_counter, other_counter, consecutive_frames, event):
    try:
        cam, _ = get_source(args)
        ret, img = cam.read()
    except Exception as e:
        queue.put(None)
        event.set()
        print(f"錯誤：無法連接攝影機: {e}")
        return

    width, height, width_height = resize(img, args.resize, args.resolution)
    logging.debug(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    output_video = None
    frame = 0
    t0 = time.time()

    while not event.is_set():
        # 若採用單攝影機，直接處理即可
        ret, img = cam.read()
        frame += 1
        self_counter.value += 1
        if img is None:
            print('無更多影像')
            if not event.is_set():
                event.set()
            break

        img = cv2.resize(img, (width, height))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        keypoint_sets, bb_list, width_height = processor_singleton.single_image(img)
        if not args.coco_points:
            anns = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
            ubboxes = [(np.asarray([width, height]) * np.asarray(ann[1])).astype('int32')
                       for ann in anns]
            lbboxes = [(np.asarray([width, height]) * np.asarray(ann[2])).astype('int32')
                       for ann in anns]
            bbox_list = [(np.asarray([width, height]) * np.asarray(box)).astype('int32') for box in bb_list]
            uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
            lhist_list = [get_hist(img, bbox) for bbox in lbboxes]
            keypoint_sets = [{"keypoints": keyp[0], "up_hist": uh, "lo_hist": lh, "time": time.time(), "box": box}
                             for keyp, uh, lh, box in zip(anns, uhist_list, lhist_list, bbox_list)]
            cv2.polylines(img, ubboxes, True, (255, 0, 0), 2)
            cv2.polylines(img, lbboxes, True, (0, 255, 0), 2)
            for box in bbox_list:
                cv2.rectangle(img, tuple(box[0]), tuple(box[1]), (0, 0, 255), 2)

        dict_vis = {"img": img, "keypoint_sets": keypoint_sets, "width": width, "height": height, 
                    "vis_keypoints": args.joints, "vis_skeleton": args.skeleton, "CocoPointsOn": args.coco_points,
                    "tagged_df": {"text": f"Avg FPS: {frame // (time.time()-t0)}, Frame: {frame}", "color": [0, 0, 0]}}
        queue.put(dict_vis)

    queue.put(None)
    return

def alg2_sequential(queues, argss, consecutive_frames, event):
    model = LSTMModel(h_RNN=48, h_RNN_layers=2, drop_p=0.1, num_classes=7)
    try:
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_script_dir, 'model', 'lstm_weights.sav')
        model.load_state_dict(torch.load(model_path, map_location=argss[0].device))
        model.eval()
    except Exception as e:
        print(f"錯誤：無法加載LSTM模型: {e}")
        return

    output_video = None
    t0 = time.time()
    ip_set = []
    lstm_set = []
    num_matched = 0
    alarm_triggered = False  # 用來避免連續觸發警報

    window_name = argss[0].video if isinstance(argss[0].video, str) else 'Webcam'
    cv2.namedWindow(window_name)

    while True:
        if not queues[0].empty():
            dict_frame = queues[0].get()
            if dict_frame is None:
                if not event.is_set():
                    event.set()
                break

            # 單攝影機情況：比對並更新 ip_set 與 lstm_set
            num_matched, new_num, _ = match_ip(ip_set, dict_frame["keypoint_sets"], lstm_set, num_matched, consecutive_frames)
            valid_idxs, prediction = get_all_features(ip_set, lstm_set, model)
            predicted_activity = activity_dict.get(prediction+5, "Unknown")
            dict_frame["tagged_df"]["text"] += f" Pred: {predicted_activity}"

            # 判斷跌倒狀態，若偵測到 "FALL" 或 "FALL Warning" 則發出警報
            if predicted_activity in ["FALL", "FALL Warning"]:
                if not alarm_triggered:
                    print("ALARM: Fall detected!")
                    os.system("afplay /System/Library/Sounds/Glass.aiff")
                    alarm_triggered = True
            else:
                alarm_triggered = False

            img, output_video = show_tracked_img(dict_frame, ip_set, num_matched, output_video, argss[0])
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) == 27:
                if not event.is_set():
                    event.set()
                break

    cv2.destroyAllWindows()
    del model
    return

def get_all_features(ip_set, lstm_set, model):
    valid_idxs = []
    invalid_idxs = []
    predictions = [15] * len(ip_set)  # 15 is the tag for None

    for i, ips in enumerate(ip_set):
        last1 = None
        last2 = None
        for j in range(-2, -1 * DEFAULT_CONSEC_FRAMES - 1, -1):
            if ips[j] is not None:
                if last1 is None:
                    last1 = j
                elif last2 is None:
                    last2 = j
        if ips[-1] is None:
            invalid_idxs.append(i)
        else:
            ips[-1]["features"] = {}
            ips[-1]["features"]["height_bbox"] = get_height_bbox(ips[-1])
            ips[-1]["features"]["ratio_bbox"] = FEATURE_SCALAR["ratio_bbox"] * get_ratio_bbox(ips[-1])
            body_vector = ips[-1]["keypoints"]["N"] - ips[-1]["keypoints"]["B"]
            ips[-1]["features"]["angle_vertical"] = FEATURE_SCALAR["angle_vertical"] * get_angle_vertical(body_vector)
            ips[-1]["features"]["log_angle"] = FEATURE_SCALAR["log_angle"] * np.log(1 + np.abs(ips[-1]["features"]["angle_vertical"]))

            if last1 is None:
                invalid_idxs.append(i)
            else:
                ips[-1]["features"]["re"] = FEATURE_SCALAR["re"] * get_rot_energy(ips[last1], ips[-1])
                ips[-1]["features"]["ratio_derivative"] = FEATURE_SCALAR["ratio_derivative"] * get_ratio_derivative(ips[last1], ips[-1])
                if last2 is None:
                    invalid_idxs.append(i)
                else:
                    ips[-1]["features"]["gf"] = get_gf(ips[last2], ips[last1], ips[-1])
                    valid_idxs.append(i)

        xdata = []
        if ips[-1] is None:
            if last1 is None:
                xdata = [0] * len(FEATURE_LIST)
            else:
                for feat in FEATURE_LIST[:FRAME_FEATURES]:
                    xdata.append(ips[last1]["features"].get(feat, 0))
                xdata += [0] * (len(FEATURE_LIST) - FRAME_FEATURES)
        else:
            for feat in FEATURE_LIST:
                xdata.append(ips[-1]["features"].get(feat, 0))

        xdata = torch.Tensor(xdata).view(-1, 1, 5)
        outputs, lstm_set[i][0] = model(xdata, lstm_set[i][0])
        if i == 0:
            prediction = torch.max(outputs.data, 1)[1][0].item()
            predictions[i] = prediction

    return valid_idxs, predictions[0] if len(predictions) > 0 else 15

def show_tracked_img(img_dict, ip_set, num_matched, output_video, args):
    img = img_dict["img"]
    tagged_df = img_dict["tagged_df"]
    # 取每個 track 中最後一個人像的關鍵點資料
    keypoints_frame = [person[-1] for person in ip_set]
    img = visualise_tracking(img=img, keypoint_sets=keypoints_frame, width=img_dict["width"],
                             height=img_dict["height"], num_matched=num_matched,
                             vis_keypoints=img_dict["vis_keypoints"], vis_skeleton=img_dict["vis_skeleton"],
                             CocoPointsOn=args.coco_points)
    img = write_on_image(img=img, text=tagged_df["text"], color=tagged_df["color"])
    if output_video is None:
        if args.save_output:
            vidname = args.video if isinstance(args.video, str) else 'webcam.avi'
            filename = 'out_' + vidname
            output_video = cv2.VideoWriter(filename=filename,
                                           fourcc=cv2.VideoWriter_fourcc(*'MP42'),
                                           fps=args.fps, frameSize=img.shape[:2][::-1])
            logging.debug(f'Saving output video at {filename} with {args.fps} FPS')
    else:
        output_video.write(img)
    return img, output_video

def get_hist(img, bbox, nbins=3):
    if not np.any(bbox):
        return None
    mask = Image.new('L', (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(list(bbox.flatten()), outline=1, fill=1)
    mask = np.array(mask)
    hist = cv2.calcHist([img], [0, 1], mask, [nbins, 2*nbins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
    return hist

@unique
class CocoPart(IntEnum):
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16

SKELETON_CONNECTIONS_COCO = [(0, 1, (210, 182, 247)), (0, 2, (127, 127, 127)), (1, 2, (194, 119, 227)),
                             (1, 3, (199, 199, 199)), (2, 4, (34, 189, 188)), (3, 5, (141, 219, 219)),
                             (4, 6, (207, 190, 23)), (5, 6, (150, 152, 255)), (5, 7, (189, 103, 148)),
                             (5, 11, (138, 223, 152)), (6, 8, (213, 176, 197)), (6, 12, (40, 39, 214)),
                             (7, 9, (75, 86, 140)), (8, 10, (148, 156, 196)), (11, 12, (44, 160, 44)),
                             (11, 13, (232, 199, 174)), (12, 14, (120, 187, 255)), (13, 15, (180, 119, 31)),
                             (14, 16, (14, 127, 255))]
SKELETON_CONNECTIONS_5P = [('H', 'N', (210, 182, 247)), ('N', 'B', (210, 182, 247)), 
                           ('B', 'KL', (210, 182, 247)), ('B', 'KR', (210, 182, 247)), 
                           ('KL', 'KR', (210, 182, 247))]
COLOR_ARRAY = [(210, 182, 247), (127, 127, 127), (194, 119, 227), (199, 199, 199), (34, 189, 188),
               (141, 219, 219), (207, 190, 23), (150, 152, 255), (189, 103, 148), (138, 223, 152)]
UNMATCHED_COLOR = (180, 119, 31)
activity_dict = {1.0: "Falling forward using hands",
                 2.0: "Falling forward using knees",
                 3: "Falling backwards",
                 4: "Falling sideward",
                 5: "FALL",
                 6: "Normal",
                 7: "Normal",
                 8: "Normal",
                 9: "Normal",
                 10: "Normal",
                 11: "Normal",
                 12: "FALL Warning",
                 20: "None"}

def write_on_image(img: np.ndarray, text: str, color: List) -> np.ndarray:
    h, w = img.shape[:2]
    lines = text.split('\n')
    # 計算背景區高度 (每行30像素，加上額外邊距)
    bg_height = 10 + 30 * len(lines)
    # 建立overlay，在影像上畫上半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bg_height), (50, 50, 50), -1)
    alpha = 0.6  # 調整透明度
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # 繪製文字，每行左上角對齊，並稍微有點邊距
    for i, line in enumerate(lines):
        pos = (10, 25 + i * 30)
        cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img

def visualise(img: np.ndarray, keypoint_sets: List, width: int, height: int,
              vis_keypoints: bool = False, vis_skeleton: bool = False, CocoPointsOn: bool = False) -> np.ndarray:
    SKELETON_CONNECTIONS = SKELETON_CONNECTIONS_COCO if CocoPointsOn else SKELETON_CONNECTIONS_5P
    if vis_keypoints or vis_skeleton:
        for keypoints in keypoint_sets:
            if not CocoPointsOn:
                keypoints = keypoints["keypoints"]
            if vis_skeleton:
                for p1i, p2i, color in SKELETON_CONNECTIONS:
                    if keypoints[p1i] is None or keypoints[p2i] is None:
                        continue
                    p1 = (int(keypoints[p1i][0] * width), int(keypoints[p1i][1] * height))
                    p2 = (int(keypoints[p2i][0] * width), int(keypoints[p2i][1] * height))
                    if p1 == (0, 0) or p2 == (0, 0):
                        continue
                    cv2.line(img=img, pt1=p1, pt2=p2, color=color, thickness=3)
    return img

def visualise_tracking(img: np.ndarray, keypoint_sets: List, width: int, height: int, num_matched: int,
                       vis_keypoints: bool = False, vis_skeleton: bool = False, CocoPointsOn: bool = False) -> np.ndarray:
    SKELETON_CONNECTIONS = SKELETON_CONNECTIONS_COCO if CocoPointsOn else SKELETON_CONNECTIONS_5P
    if vis_keypoints or vis_skeleton:
        for i, keypoints in enumerate(keypoint_sets):
            if keypoints is None:
                continue
            if not CocoPointsOn:
                keypoints = keypoints["keypoints"]
            if vis_skeleton:
                for p1i, p2i, color in SKELETON_CONNECTIONS:
                    if keypoints[p1i] is None or keypoints[p2i] is None:
                        continue
                    p1 = (int(keypoints[p1i][0] * width), int(keypoints[p1i][1] * height))
                    p2 = (int(keypoints[p2i][0] * width), int(keypoints[p2i][1] * height))
                    if p1 == (0, 0) or p2 == (0, 0):
                        continue
                    color = COLOR_ARRAY[i % 10] if i < num_matched else UNMATCHED_COLOR
                    cv2.line(img=img, pt1=p1, pt2=p2, color=color, thickness=3)
    return img

class Processor(object):
    def __init__(self, width_height, args):
        self.width_height = width_height
        self.model_cpu, _ = openpifpaf.network.Factory().factory()
        self.model = self.model_cpu.to(args.device)
        self.processor = openpifpaf.decoder.factory(self.model_cpu.head_metas)
        self.device = args.device

    def get_bb(self, kp_set):
        bb_list = []
        for i in range(kp_set.shape[0]):
            x = kp_set[i, :15, 0]
            y = kp_set[i, :15, 1]
            v = kp_set[i, :15, 2]
            if not np.any(v > 0):
                return None
            x1, x2 = np.min(x[v > 0]), np.max(x[v > 0])
            y1, y2 = np.min(y[v > 0]), np.max(y[v > 0])
            if x2 - x1 < 5.0 / self.width_height[0]:
                x1 -= 2.0 / self.width_height[0]
                x2 += 2.0 / self.width_height[0]
            if y2 - y1 < 5.0 / self.width_height[1]:
                y1 -= 2.0 / self.width_height[1]
                y2 += 2.0 / self.width_height[1]
            bb_list.append(((x1, y1), (x2, y2)))
        return bb_list

    @staticmethod
    def keypoint_sets(annotations):
        keypoint_sets = [ann.data for ann in annotations]
        if not keypoint_sets:
            return np.zeros((0, 17, 3))
        return np.array(keypoint_sets)

    def single_image(self, image):
        im = PIL.Image.fromarray(image)
        target_wh = self.width_height
        if (im.size[0] > im.size[1]) != (target_wh[0] > target_wh[1]):
            target_wh = (target_wh[1], target_wh[0])
        if im.size[0] != target_wh[0] or im.size[1] != target_wh[1]:
            im = im.resize(target_wh, PIL.Image.BICUBIC)
        width_height = im.size
        preprocess = openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.CenterPadTight(16),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])
        processed_image = openpifpaf.datasets.PilImageList([im], preprocess=preprocess)[0][0]
        all_fields = self.processor.batch(self.model, torch.unsqueeze(processed_image.float(), 0), device=self.device)[0]
        keypoint_sets = self.keypoint_sets(all_fields)
        keypoint_sets[:, :, 0] /= processed_image.shape[2]
        keypoint_sets[:, :, 1] /= processed_image.shape[1]
        bboxes = self.get_bb(keypoint_sets)
        return keypoint_sets, bboxes, width_height

def pop_and_add(l, val, max_length):
    if len(l) == max_length:
        l.pop(0)
    l.append(val)

def last_ip(ips):
    for i, ip in enumerate(reversed(ips)):
        if ip is not None:
            return ip

def last_valid_hist(ips):
    for ip in reversed(ips):
        if ip is not None and ip.get("up_hist") is not None:
            return ip

def dist(ip1, ip2):
    ip1 = ip1["keypoints"]
    ip2 = ip2["keypoints"]
    return np.sqrt(np.sum((ip1['N'] - ip2['N'])**2 + (ip1['B'] - ip2['B'])**2))

def move_figure(f, x, y):
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        f.canvas.manager.window.move(x, y)

def valid_candidate_hist(ip):
    return ip is not None and ip.get("up_hist") is not None

def dist_hist(ips1, ips2):
    ip1 = last_valid_hist(ips1)
    ip2 = last_valid_hist(ips2)
    uhist1 = ip1["up_hist"]
    uhist2 = ip2["up_hist"]
    return np.sum(np.absolute(uhist1 - uhist2))

def get_kp(kp):
    threshold1 = 5e-3
    inv_pend = {}
    numx = (kp[CocoPart.LEar][2]*kp[CocoPart.LEar][0] + kp[CocoPart.LEye][2]*kp[CocoPart.LEye][0] +
            kp[CocoPart.REye][2]*kp[CocoPart.REye][0] + kp[CocoPart.REar][2]*kp[CocoPart.REar][0])
    numy = (kp[CocoPart.LEar][2]*kp[CocoPart.LEar][1] + kp[CocoPart.LEye][2]*kp[CocoPart.LEye][1] +
            kp[CocoPart.REye][2]*kp[CocoPart.REye][1] + kp[CocoPart.REar][2]*kp[CocoPart.REar][1])
    den = kp[CocoPart.LEar][2] + kp[CocoPart.LEye][2] + kp[CocoPart.REye][2] + kp[CocoPart.REar][2]
    inv_pend['H'] = None if den < HEAD_THRESHOLD else np.array([numx/den, numy/den])
    if all([kp[CocoPart.LShoulder], kp[CocoPart.RShoulder],
            kp[CocoPart.LShoulder][2] > threshold1, kp[CocoPart.RShoulder][2] > threshold1]):
        inv_pend['N'] = np.array([(kp[CocoPart.LShoulder][0] + kp[CocoPart.RShoulder][0]) / 2,
                                  (kp[CocoPart.LShoulder][1] + kp[CocoPart.RShoulder][1]) / 2])
    else:
        inv_pend['N'] = None
    if all([kp[CocoPart.LHip], kp[CocoPart.RHip],
            kp[CocoPart.LHip][2] > threshold1, kp[CocoPart.RHip][2] > threshold1]):
        inv_pend['B'] = np.array([(kp[CocoPart.LHip][0] + kp[CocoPart.RHip][0]) / 2,
                                  (kp[CocoPart.LHip][1] + kp[CocoPart.RHip][1]) / 2])
    else:
        inv_pend['B'] = None
    inv_pend['KL'] = np.array([kp[CocoPart.LKnee][0], kp[CocoPart.LKnee][1]]) if kp[CocoPart.LKnee] is not None and kp[CocoPart.LKnee][2] > threshold1 else None
    inv_pend['KR'] = np.array([kp[CocoPart.RKnee][0], kp[CocoPart.RKnee][1]]) if kp[CocoPart.RKnee] is not None and kp[CocoPart.RKnee][2] > threshold1 else None
    if inv_pend['B'] is not None:
        if inv_pend['N'] is not None:
            height = np.linalg.norm(inv_pend['N'] - inv_pend['B'], 2)
            LS, RS = extend_vector(np.asarray(kp[CocoPart.LShoulder][:2]),
                                   np.asarray(kp[CocoPart.RShoulder][:2]), height/4)
            LB, RB = extend_vector(np.asarray(kp[CocoPart.LHip][:2]),
                                   np.asarray(kp[CocoPart.RHip][:2]), height/3)
            ubbox = (LS, RS, RB, LB)
            lbbox = (LB, RB, inv_pend['KR'], inv_pend['KL']) if inv_pend['KL'] is not None and inv_pend['KR'] is not None else ([0, 0], [0, 0])
        else:
            ubbox = ([0, 0], [0, 0])
            lbbox = (np.array(kp[CocoPart.LHip][:2]), np.array(kp[CocoPart.RHip][:2]),
                     inv_pend['KR'], inv_pend['KL']) if inv_pend['KL'] is not None and inv_pend['KR'] is not None else ([0, 0], [0, 0])
    else:
        ubbox = ([0, 0], [0, 0])
        lbbox = ([0, 0], [0, 0])
    return inv_pend, ubbox, lbbox

def extend_vector(p1, p2, l):
    p1 += (p1 - p2) * l / (2 * np.linalg.norm((p1 - p2), 2))
    p2 -= (p1 - p2) * l / (2 * np.linalg.norm((p1 - p2), 2))
    return p1, p2

def get_angle(v0, v1):
    return np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

def is_valid(ip):
    ip = ip["keypoints"]
    return (ip['B'] is not None and ip['N'] is not None and ip['H'] is not None)

def get_rot_energy(ip0, ip1):
    t = ip1["time"] - ip0["time"]
    ip0 = ip0["keypoints"]
    ip1 = ip1["keypoints"]
    m1, m2 = 1, 5
    N1 = ip1['N'] - ip1['B']
    N0 = ip0['N'] - ip0['B']
    d2sq = N1.dot(N1)
    w2sq = (get_angle(N0, N1) / t) ** 2
    energy = m2 * d2sq * w2sq
    H1 = ip1['H'] - ip1['B']
    H0 = ip0['H'] - ip0['B']
    d1sq = H1.dot(H1)
    w1sq = (get_angle(H0, H1) / t) ** 2
    energy += m1 * d1sq * w1sq
    den = m1 * d1sq + m2 * d2sq
    return energy / (2 * den) if den != 0 else 0

def get_angle_vertical(v):
    return np.math.atan2(-v[0], -v[1])

def get_gf(ip0, ip1, ip2):
    t1 = ip1["time"] - ip0["time"]
    t2 = ip2["time"] - ip1["time"]
    ip0 = ip0["keypoints"]
    ip1 = ip1["keypoints"]
    ip2 = ip2["keypoints"]
    m1, m2, g = 1, 15, 10
    H2 = ip2['H'] - ip2['N']
    H1 = ip1['H'] - ip1['N']
    H0 = ip0['H'] - ip0['N']
    d1 = np.sqrt((H1).dot(H1))
    theta_1_plus_2_2 = get_angle_vertical(H2)
    theta_1_plus_2_1 = get_angle_vertical(H1)
    theta_1_plus_2_0 = get_angle_vertical(H0)
    N2 = ip2['N'] - ip2['B']
    N1 = ip1['N'] - ip1['B']
    N0 = ip0['N'] - ip0['B']
    d2 = np.sqrt((N1).dot(N1))
    theta_2_2 = get_angle_vertical(N2)
    theta_2_1 = get_angle_vertical(N1)
    theta_2_0 = get_angle_vertical(N0)
    theta1 = theta_1_plus_2_1 - theta_2_1
    del_theta1 = 0.5 * ((get_angle(H0, H1)) / t1 + (get_angle(H1, H2)) / t2)
    del_theta2 = 0.5 * ((get_angle(N0, N1)) / t1 + (get_angle(N1, N2)) / t2)
    doubledel_theta1 = (get_angle(H1, H2) - get_angle(H0, H1)) / (0.5 * (t1 + t2))
    doubledel_theta2 = (get_angle(N1, N2) - get_angle(N0, N1)) / (0.5 * (t1 + t2))
    d1 = d1 / d2 if d2 != 0 else 0
    Q_RD1 = m1 * d1 * (doubledel_theta1 ** 2) + (m1 * d1 * d1 + m1 * d1 * np.cos(theta1)) * doubledel_theta2 - m1 * g * np.sin(theta1 + theta_2_1)
    return Q_RD1

def get_height_bbox(ip):
    bbox = ip["box"]
    diff_box = bbox[1] - bbox[0]
    return diff_box[1]

def get_ratio_bbox(ip):
    bbox = ip["box"]
    diff_box = bbox[1] - bbox[0]
    if diff_box[1] == 0:
        diff_box[1] += 1e5 * diff_box[0]
    return diff_box[0] / diff_box[1]

def get_ratio_derivative(ip0, ip1):
    time_diff = ip1["time"] - ip0["time"]
    diff_box = ip1["features"]["ratio_bbox"] - ip0["features"]["ratio_bbox"]
    return diff_box / time_diff if time_diff != 0 else 0

def match_ip(ip_set, new_ips, lstm_set, num_matched, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    len_ip_set = len(ip_set)
    added = [False] * len_ip_set
    new_len_ip_set = len_ip_set
    for new_ip in new_ips:
        if not is_valid(new_ip):
            continue
        cmin = [MIN_THRESH, -1]
        for i in range(len_ip_set):
            last = last_ip(ip_set[i])
            if last is None:
                continue
            if not added[i] and dist(last, new_ip) < cmin[0]:
                cmin = [dist(last, new_ip), i]
        if cmin[1] == -1:
            ip_set.append([None for _ in range(consecutive_frames - 1)] + [new_ip])
            lstm_set.append([None, 0, 0, 0])
            new_len_ip_set += 1
        else:
            added[cmin[1]] = True
            pop_and_add(ip_set[cmin[1]], new_ip, consecutive_frames)
    removed_indx = []
    for i in range(len(added)):
        if not added[i]:
            pop_and_add(ip_set[i], None, consecutive_frames)
        if ip_set[i] == [None for _ in range(consecutive_frames)]:
            if i < num_matched:
                num_matched -= 1
            new_len_ip_set -= 1
            removed_indx.append(i)
    for i in sorted(removed_indx, reverse=True):
        ip_set.pop(i)
        lstm_set.pop(i)
    return num_matched, new_len_ip_set, removed_indx

class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, h_RNN_layers=2, h_RNN=256, drop_p=0.2, num_classes=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.h_RNN_layers = h_RNN_layers
        self.h_RNN = h_RNN
        self.drop_p = drop_p if h_RNN_layers >= 2 else 0
        self.num_classes = num_classes
        self.LSTM = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            dropout=self.drop_p,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.h_RNN, self.num_classes)

    def forward(self, x, h_s=None):
        self.LSTM.flatten_parameters()
        RNN_out, h_s = self.LSTM(x, h_s)
        out = self.fc1(RNN_out[:, -1, :])
        return out, h_s

class FallDetector:
    def __init__(self, t=DEFAULT_CONSEC_FRAMES):
        self.consecutive_frames = t
        self.args = self.cli()

    def cli(self):
        parser = argparse.ArgumentParser(
            description="跌倒偵測",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        openpifpaf.network.Factory.cli(parser)
        openpifpaf.decoder.cli(parser)
        parser.add_argument('--resolution', default=0.4, type=float,
                            help='Resolution prescale factor, will be rounded to multiples of 16.')
        parser.add_argument('--resize', default=None, type=str,
                            help='Force input image resize. Example: WIDTHxHEIGHT')
        parser.add_argument('--num_cams', default=1, type=int,
                            help='Number of Cameras. (僅支援 1 支援)')
        parser.add_argument('--video', default=None, type=str,
                            help='忽略影片功能，使用 webcam 即時辨識時不指定')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='開啟 debug 訊息')
        parser.add_argument('--disable_cuda', default=False, action='store_true',
                            help='停用 CUDA, 強制使用 CPU')
        vis_args = parser.add_argument_group('Visualisation')
        vis_args.add_argument('--plot_graph', default=False, action='store_true',
                              help='顯示特徵圖形 (目前未使用)')
        vis_args.add_argument('--joints', default=True, action='store_true',
                              help='在影片中畫出關節點')
        vis_args.add_argument('--skeleton', default=True, action='store_true',
                              help='在影片中畫出骨架')
        vis_args.add_argument('--coco_points', default=False, action='store_true',
                              help='顯示 COCO 點位')
        vis_args.add_argument('--save_output', default=False, action='store_true',
                              help='儲存辨識結果影片')
        vis_args.add_argument('--fps', default=18, type=int,
                              help='輸出影片的 FPS')
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
        args.force_complete_pose = True
        args.instance_threshold = 0.2
        args.seed_threshold = 0.5
        args.device = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
        args.pin_memory = not args.disable_cuda and torch.cuda.is_available()
        if args.checkpoint is None:
            args.checkpoint = 'shufflenetv2k16w'
        openpifpaf.decoder.configure(args)
        openpifpaf.network.Factory.configure(args)
        return args

    def begin(self):
        print('開始...')
        e = mp.Event()
        queues = [mp.Queue()]
        counter1 = mp.Value('i', 0)
        counter2 = mp.Value('i', 0)
        argss = [copy.deepcopy(self.args)]
        # 若未指定影片來源，默認使用 webcam
        if self.args.video is None:
            argss[0].video = 0
        process1 = mp.Process(target=extract_keypoints_parallel,
                              args=(queues[0], argss[0], counter1, counter2, self.consecutive_frames, e))
        process1.start()
        process2 = mp.Process(target=alg2_sequential, args=(queues, argss, self.consecutive_frames, e))
        process2.start()
        process1.join()
        process2.join()
        print('結束...')
        return

if __name__ == "__main__":
    f = FallDetector()
    f.begin()