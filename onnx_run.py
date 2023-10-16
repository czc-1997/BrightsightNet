import torch.onnx
import onnxruntime

import torch
import torchvision
import torch.optim
import os

import numpy as np

from PIL import Image




# Define the path for saving the ONNX model
onnx_path = './enhanceNet.onnx'


# Load the ONNX model with ONNX Runtime
ort_session = onnxruntime.InferenceSession(onnx_path)

def lowlight_inference(image_path, ort_session):
    # Load and preprocess the input image
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)

    # Perform inference using ONNX Runtime
    ort_inputs = {'input': data_lowlight.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    enhance_image = torch.from_numpy(ort_outs[0])

    # Save the result
    result_path = image_path.replace('test_sample', 'final').replace('\\', '/')
    if not os.path.exists(result_path.replace('/' + result_path.split("/")[-1], '')):
        os.makedirs(result_path.replace('/' + result_path.split("/")[-1], ''))

    torchvision.utils.save_image(torch.cat([data_lowlight, enhance_image], dim=-1), result_path)

if __name__ == '__main__':
    # Test the ONNX model on sample images
    filePath = r'data/test_sample/'
    file_list = os.listdir(filePath)
    import time
    for image in file_list:
        t1 = time.time()
        print(image)
        lowlight_inference(os.path.join(filePath + image), ort_session)
        print(time.time()-t1)
