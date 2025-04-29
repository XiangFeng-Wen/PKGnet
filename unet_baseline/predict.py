import os
import time

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet

weights_path = "./save_weights/best_model.pth"
img_path = r"/home/wxf/project/Dataset/myBreaDM/val/all_images/013.jpg"
roi_mask_path = r"/home/wxf/project/Dataset/myBreaDM/val/all_manual/001.png"

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

#  18_23_M_SER_1005_1F  18_33_M_SER_1002_2D  18_13_B_SUB3_32
def main():
    classes = 1  # exclude background
    
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    # For single-channel (grayscale) images, we use a single mean and std value
    mean = (0.381,)  # Adjusted for grayscale image
    std = (0.079,)   # Adjusted for grayscale image

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=1, num_classes=classes + 1, base_c=32)
    
    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert("L")  # Ensure image is single-channel (grayscale)
    
    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])  # Adapted for grayscale
    img = data_transform(original_img)
    
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # Shape becomes (1, 1, H, W)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 1, img_height, img_width), device=device)
        model(init_img)  # Initialize model with a dummy input

        t_start = time_synchronized()
        output = model(img.to(device))  # Perform actual inference
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # # 将不敢兴趣的区域像素设置成0(黑色)
        # prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("./output/unet_predict.png")


if __name__ == '__main__':
    main()
