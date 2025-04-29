import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import jaccard_score
from tqdm import tqdm

def dice_coefficient(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)  # 获取预测的类别
    dice = 0.0
    for cls in range(1, num_classes):  # 假设背景为0，不计入
        pred_mask = (pred == cls).float()
        target_mask = (target == cls).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        if union == 0:  # 防止除以零
            continue
        dice += 2.0 * intersection / union
    return dice / (num_classes - 1)

def jaccard_index(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).cpu().numpy().ravel()
    target = target.cpu().numpy().ravel()
    iou = jaccard_score(target, pred, average='macro', labels=range(1, num_classes))
    return iou


def evaluate(model, data_loader, device, num_classes):
    model.eval()  # 设置模型为评估模式
    total_dice = 0.0
    with torch.no_grad():  # 关闭梯度计算
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks, num_classes)
            total_dice += dice.item()
            iou = jaccard_index(outputs, masks, num_classes)
            total_iou += iou

    avg_dice = total_dice / len(data_loader)
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    avg_iou = total_iou / len(data_loader)
    print(f"Average iou Coefficient: {avg_iou:.4f}")
    return avg_dice

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载验证数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_dataset = CustomSegmentationDataset(root='./dataset/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 加载训练好的模型
    model = UNet(num_classes=2)  # 假设二分类（背景+前景）
    model.load_state_dict(torch.load('./save_weights/best_model.pth', map_location=device))
    model.to(device)
    
    num_classes = 2  # 根据实际分类数设置
    avg_dice = evaluate(model, val_loader, device, num_classes)
    # print(f"Final Dice Coefficient on Validation Set: {avg_dice:.4f}")


if __name__ == '__main__':
    main()

