import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import mediapipe as mp

from net import Net
from option import Options
import utils
from utils import StyleLoader

def run_demo(args, mirror=False, segment_human=True):
	# 初始化 mediapipe selfie segmentation
	mp_selfie_segmentation = mp.solutions.selfie_segmentation
	selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

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
	while True:
		# read frame
		ret_val, img = cam.read()
		if mirror:
			img = cv2.flip(img, 1)
		cimg = img.copy()

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
		style_v = style_loader.get(0)
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
