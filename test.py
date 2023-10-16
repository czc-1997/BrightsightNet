import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader

import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from Ours.BrightsightNet import enhanceNet


def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight) / 255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2, 0, 1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = enhanceNet().cuda()
	DCE_net.eval()
	DCE_net.load_state_dict(torch.load('./iters100000.pth'))
	start = time.time()
	enhace1, enhance_image, _, _ = DCE_net(data_lowlight)
	print(enhance_image)

	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('test_sample', 'final')
	image_path = image_path.replace('\\', '/')
	result_path = image_path
	if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
		os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
	# resut = torch.stack([data_lowlight, enhanced_image])
	# print(resut.size())
	shape = data_lowlight.size()

	torchvision.utils.save_image(torch.cat([data_lowlight, enhance_image], dim=-1), result_path)


if __name__ == '__main__':
	# test_images
	with torch.no_grad():
		filePath = r'data/test_sample/'

		file_list = os.listdir(filePath)

		for image in file_list:
			print(image)
			lowlight(os.path.join(filePath + image))



