import torch
from torch import nn
import numpy as np
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes) #用于记录每个类别的混淆矩阵数据。
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ") #用于每 100 个 batch 打印一次日志。
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        mat = confmat.mat.cpu().numpy()  # 获取混淆矩阵数值

    eps = 1e-6  # 防止除零的小量
    total = mat.sum()
    class_metrics = []

    # 计算全局准确率
    global_accuracy = np.diag(mat).sum() / total if total != 0 else 0.0

    # 遍历每个类别计算指标
    for cls_idx in range(num_classes):
        tp = mat[cls_idx, cls_idx]
        fp = mat[:, cls_idx].sum() - tp
        fn = mat[cls_idx, :].sum() - tp
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        # dice_t = 2 * tp / (2 * tp + fp + fn + eps)
        
        class_metrics.append({
            "precision": precision,
            "recall": recall,
            "iou": iou,
            # "dice": dice_t
        })

    # 计算平均指标
    mean_metrics = {
        "mprecision": np.mean([m["precision"] for m in class_metrics]),
        "mrecall": np.mean([m["recall"] for m in class_metrics]),
        "miou": np.mean([m["iou"] for m in class_metrics]),
        # "mdice": np.mean([m["dice"] for m in class_metrics])
    }

    return {
        "dice": dice.value.item(),
        "confusion_matrix": confmat,
        "global_accuracy": global_accuracy,
        "class_metrics": class_metrics,
        "mean_metrics": mean_metrics
    }

    #return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None
    # print(len(data_loader))
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
