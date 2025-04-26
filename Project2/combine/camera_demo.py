import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import mediapipe as mp
import joblib
from ultralytics import YOLO

from net import Net
from option import Options
import utils
from utils import StyleLoader

def compute_distances(landmarks):
    # Define pairs for distance calculation
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8),
             (0, 9), (0, 10), (0, 11), (0, 12),
             (0, 13), (0, 14), (0, 15), (0, 16),
             (0, 17), (0, 18), (0, 19), (0, 20),
             (4, 8), (8, 12), (12, 16), (16, 20)]

    distances = []
    reference_distance = np.linalg.norm(
        np.array([landmarks.landmark[0].x, landmarks.landmark[0].y]) -
        np.array([landmarks.landmark[9].x, landmarks.landmark[9].y])
    )

    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2)
        distances.append(distance/reference_distance)  # Normalize the distance

    return distances

def run_demo(args, mirror=False, segment_human=False, detect_gesture=False, detect_person=False):
	# 初始化 mediapipe selfie segmentation
	mp_selfie_segmentation = mp.solutions.selfie_segmentation
	selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

	# 初始化手勢辨識
	mp_hands = mp.solutions.hands
	hands = mp_hands.Hands()
	
	# 載入手勢模型
	model_filename = "models/svm_model.pkl"
	clf = joblib.load(model_filename)
	scaler_filename = "models/scaler.pkl"
	scaler = joblib.load(scaler_filename)
	
	# 載入標籤
	label_file = "models/labels.txt"
	with open(label_file, 'r') as f:
		labels = f.readlines()
	labels = [label.strip() for label in labels]

	# 初始化 YOLO 模型
	try:
		print("正在載入 YOLO 模型...")
		yolo_model = YOLO('./models/yolov8n.pt')
		print("YOLO 模型載入成功")
	except Exception as e:
		print(f"YOLO 模型載入失敗: {str(e)}")
		detect_person = False

	style_model = Net(ngf=args.ngf)
	model_dict = torch.load(args.model)
	model_dict_clone = model_dict.copy()
	for key, value in model_dict_clone.items():
		if key.endswith(('running_mean', 'running_var')):
			del model_dict[key]
	style_model.load_state_dict(model_dict, False)
	style_model.eval()
	if args.cuda:
		style_loader = StyleLoader(args.style_folder, args.style_size)
		style_model.cuda()
	else:
		style_loader = StyleLoader(args.style_folder, args.style_size, False)

	# Define the codec and create VideoWriter object
	height = args.demo_size
	width = int(4.0/3*args.demo_size)
	swidth = int(width/4)
	sheight = int(height/4)
	if args.record:
		fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
		out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, height))
	cam = cv2.VideoCapture(0)
	cam.set(3, width)
	cam.set(4, height)
	key = 0
	idx = 0
	style_idx = 0
	last_gesture = None
	while True:
		# read frame
		ret_val, img = cam.read()
		if mirror:
    			img = cv2.flip(img, 1)
		cimg = img.copy()

		# 人數計數
		if detect_person:
			try:
				results = yolo_model(cimg, stream=True)
				person_count = 0
				for result in results:
					boxes = result.boxes
					for box in boxes:
						cls_id = int(box.cls[0])
						if yolo_model.names[cls_id] == 'person':
							person_count += 1
							x1, y1, x2, y2 = map(int, box.xyxy[0])
							cv2.rectangle(cimg, (x1, y1), (x2, y2), (0, 0, 255), 2)
							cv2.putText(cimg, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				cv2.putText(cimg, f"People Count: {person_count}", (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			except Exception as e:
				print(f"YOLO 推論失敗: {str(e)}")
				detect_person = False

		# 手勢辨識
		if detect_gesture:
			rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			hand_results = hands.process(rgb_frame)
			
			if hand_results.multi_hand_landmarks:
				for index, landmarks in enumerate(hand_results.multi_hand_landmarks):
					# 分辨左右手
					hand_label = "Right" if hand_results.multi_handedness[index].classification[0].label == "Left" else "Left"
					
					distances = compute_distances(landmarks)
					distances = scaler.transform([distances])
					
					prediction = clf.predict(distances)
					confidence = np.max(clf.predict_proba(distances))
					
					label = labels[prediction[0]]
					display_text = f"{hand_label} Hand: {label} ({confidence*100:.2f}%)"
					
					cv2.putText(cimg, display_text, (10, 30 + (index * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
					
					# 顯示手部關鍵點
					mp.solutions.drawing_utils.draw_landmarks(cimg, landmarks, mp_hands.HAND_CONNECTIONS)
					
					# 根據手勢切換風格
					if confidence > 0.8:
						if label != last_gesture:  # 只有當手勢改變時才切換
							if label == "Good":
								style_idx = (style_idx - 1) % style_loader.size()
								last_gesture = label
							elif label == "Bad":
								style_idx = (style_idx + 1) % style_loader.size()
								last_gesture = label
			else:
				last_gesture = None  # 當沒有檢測到手時重置上一次的手勢

		# 人像分割
		if segment_human:
			image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			results = selfie_segmentation.process(image_rgb)
			mask = results.segmentation_mask
			condition = mask > 0.5
			condition = condition.astype(np.uint8) * 255
			condition_3ch = cv2.merge([condition, condition, condition])
		else:
			condition_3ch = np.zeros_like(img)

		# 風格轉換
		img = np.array(img).transpose(2, 0, 1)
		style_v = style_loader.get(style_idx)
		style_v = Variable(style_v.data)
		style_model.setTarget(style_v)

		img=torch.from_numpy(img).unsqueeze(0).float()
		if args.cuda:
			img=img.cuda()

		img = Variable(img)
		img = style_model(img)

		if args.cuda:
			simg = style_v.cpu().data[0].numpy()
			img = img.cpu().clamp(0, 255).data[0].numpy()
		else:
			simg = style_v.data.numpy()
			img = img.clamp(0, 255).data[0].numpy()
		simg = np.squeeze(simg)
		img = img.transpose(1, 2, 0).astype('uint8')
		simg = simg.transpose(1, 2, 0).astype('uint8')

		# 將風格化結果與原始影像根據人像遮罩合併
		img_original = cimg.copy()
		img_stylized = img.copy()
		if segment_human:
			result = np.where(condition_3ch > 0, img_original, img_stylized)
		else:
			result = img_stylized

		# display
		simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
		result[0:sheight,0:swidth,:]=simg
		img = np.concatenate((cimg,result),axis=1)
		cv2.imshow('MSG Demo', img)

		key = cv2.waitKey(1)
		if args.record:
			out.write(img)
		if key == 27:
			break
		elif key == ord('s'):  # 按's'鍵切換人像分割
			segment_human = not segment_human
		elif key == ord('g'):  # 按'g'鍵切換手勢辨識
			detect_gesture = not detect_gesture
		elif key == ord('w'):  # 按'w'鍵切換人數計數
			detect_person = not detect_person
		elif key == ord('a'):  # 按'a'鍵切換上一個風格
			style_idx = (style_idx - 1) % style_loader.size()
		elif key == ord('d'):  # 按'd'鍵切換下一個風格
			style_idx = (style_idx + 1) % style_loader.size()
	cam.release()
	if args.record:
		out.release()
	cv2.destroyAllWindows()

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	# run demo
	run_demo(args, mirror=True)

if __name__ == '__main__':
	main()
