import torch
import torchvision
import torch.optim
import os
import numpy as np
from PIL import Image
import time
from Ours.BrightsightNet import enhanceNet

DCE_net = enhanceNet(mode="test")
DCE_net.eval()
DCE_net.load_state_dict(torch.load('./iters100000.pth',map_location=torch.device('cpu')))
def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight)/255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.unsqueeze(0)
	enhance_image = DCE_net(data_lowlight)
	image_path = image_path.replace('test_sample','final')
	image_path = image_path.replace('\\', '/')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))



	torchvision.utils.save_image(torch.cat([ data_lowlight,enhance_image], dim=-1), result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = r'data/test_sample/'
	
		file_list = os.listdir(filePath)

		for image in file_list:
			print(image)
			lowlight(os.path.join(filePath+image))

		

