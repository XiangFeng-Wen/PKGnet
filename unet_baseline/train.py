import os
import time
import datetime

import torch

from src import deeplabv3_resnet50,UNet,fcn_resnet50,transunet,TSESNet
from src.xlstm_unet import xLSTMUNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T

import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.cuda.amp import autocast  #  显式导入autocast

#预处理类。在训练时的数据增强包括随机缩放、水平翻转、垂直翻转、随机裁剪，然后转为Tensor和标准化
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485), std=(0.229)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

#验证，调整大小到crop_size，然后同样的转换。
class SegmentationPresetEval:
    def __init__(self, crop_size,mean=(0.485), std=(0.229)):
        self.transforms = T.Compose([
            T.RandomResize(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485), std=(0.229)):
    base_size = 120
    crop_size = 100

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(crop_size,mean=mean, std=std)


def create_model(aux, num_classes, model_type="unet", pretrain=False):
    if model_type.lower() == "unet":
        model = UNet(in_channels=1, num_classes=num_classes, base_c=32)
    elif model_type.lower() == "transunet":
        model = transunet.TransUNet(in_channels=1, num_classes=num_classes, base_c=32)
    elif model_type.lower() == "deeplabv3":
        model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)
    elif model_type.lower() == "fcn":
        model = fcn_resnet50(aux=aux, num_classes=num_classes)
    elif model_type.lower() == "xlstm-unet":
        model = xLSTMUNet(in_channels=1, num_classes=num_classes, base_c=32)
    elif model_type.lower() == "tsesnet":
        model = TSESNet(in_channels=1, num_classes=num_classes, base_c=32)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if pretrain and model_type.lower() in ["deeplabv3", "fcn"]:
        weights_dict = torch.load("./fcn_resnet50_coco.pth", map_location='cpu')
        if num_classes != 21:
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    print(torch.__version__)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))

    #  启用TF32加速（需PyTorch 1.12+）
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32 tensor cores")

    #  优化cudnn配置
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cudnn.deterministic = False  # 允许非确定性优化

    # 数据加载优化
    num_workers = args.workers  #  使用自定义workers数

    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709)
    std = (0.127)

    # 用来保存训练以及验证过程中信息，命名包括当前模型类型和训练时间
    results_file = None
    if not args.silent:
        results_file = "./output/{}_results_{}.txt".format(
            args.model_type,
            datetime.datetime.now().strftime("%m%d-%H%M")
        )

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))
    # 智能配置num_workers
    cpu_count = os.cpu_count()
    # 根据经验，num_workers通常设置为CPU核心数的0.5-1倍较为合适
    suggested_workers = max(1, min(int(cpu_count * 0.75), batch_size * 2))
    # 考虑用户指定的workers数量
    num_workers = min([suggested_workers, args.workers if args.workers > 0 else float('inf')])

    print(f"Using {num_workers} workers for data loading, batch_size: {batch_size}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True,  # 锁页内存加速传输
                                               prefetch_factor=2,  # 预加载批次，避免内存过载
                                               persistent_workers=True,  # 保持worker进程存活
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             shuffle = False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(aux=args.aux, num_classes=num_classes, model_type=args.model_type)
    model.to(device)

    # params_to_optimize = [
    #     {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
    #     {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    # ]
    #
    # if args.aux:
    #     params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
    #     params_to_optimize.append({"params": params, "lr": args.lr * 10}) # FCN增加
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(
    #     params_to_optimize,
    #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    # )
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,                    # 建议更小的学习率（例如 1e-4）
        betas=(0.9, 0.999),            # AdamW 的动量参数（默认值）
        weight_decay=args.weight_decay, # 解耦权重衰减（关键特性）
        eps=1e-8,                       # 数值稳定性参数（可选）
        fused=True
    )

    scaler = torch.amp.GradScaler() if args.amp else None
    if args.amp:
        print(f"Mixed precision training enabled (AMP mode)")

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    
    # 记录当前运行参数,包括当前模型类型、batch_size、学习率、优化器等信息
    if not args.silent and results_file:
        with open(results_file, mode='a') as f:
            f.write(str(args) + "\n")

    # 获取训练开始时间戳，用于模型命名
    start_timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    
    print("Start training...")
    best_dice = 0.
    start_time = time.time()
    patience_counter = 0
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        # confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        # val_info = str(confmat)
        # print(val_info)
        # print(f"dice coefficient: {dice:.3f}")

        results = evaluate(model, val_loader, device, num_classes)
        val_info=str(results["confusion_matrix"])
        dice=results['dice']
        # dice=results['mean_metrics']['dice']

        print(val_info)
        print(f"dice coefficient: {dice:.3f}")

        # 打印详细指标并保存到results_file中
        print("Confusion Matrix:")
        print(f"Global Accuracy: {results['global_accuracy']:.4f}")

        print("\nPer-class Metrics:")
        for cls_idx, metrics in enumerate(results["class_metrics"]):
            print(f"Class {cls_idx}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  IoU:       {metrics['iou']:.4f}")
            # print(f"  Dice:      {metrics['dice']:.4f}")

        print("\nMean Metrics:")
        print(f"Precision: {results['mean_metrics']['mprecision']:.4f}")
        print(f"Recall:    {results['mean_metrics']['mrecall']:.4f}")
        print(f"IoU:       {results['mean_metrics']['miou']:.4f}")
        # print(f"Dice:      {results['mean_metrics']['mdice']:.4f}")

        # write into txt
        if not args.silent and results_file:
            with open(results_file, mode='a') as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n" \
                             f"dice coefficient: {dice:.3f}\n"
                f.write(train_info + val_info + "\n\n")

        # 检查是否有改进
        if dice > best_dice + args.min_delta:
            best_dice = dice
            patience_counter = 0
            
            # 保存最佳模型
            if args.save_best is True and not args.silent:
                save_file = {"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            "best_dice": best_dice}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()
                
                # 使用固定文件名保存最佳模型，使用训练开始时的时间戳
                best_model_path = f"save_weights/best_{args.model_type}_{start_timestamp}.pth"
                torch.save(save_file, best_model_path)
                
                print(f"保存新的最佳模型，Dice: {dice:.4f}，路径: {best_model_path}")
        else:
            patience_counter += 1

        # 检查是否需要早停
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered! No improvement for {args.patience} epochs.")
            break

        # 如果不是保存最佳模型且不是静默模式，则每个epoch都保存
        if args.save_best is False and not args.silent:
            save_file = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            
            # 添加当前时间戳到每个epoch的保存文件名中
            current_timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
            torch.save(save_file, f"save_weights/model_{epoch}_{current_timestamp}.pth")

        torch.cuda.empty_cache()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None, scaler=None, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ----------------- 模型参数加载 -----------------
    model_state = checkpoint['model']
    
    # 处理多GPU训练保存的参数前缀
    if all(k.startswith('module.') for k in model_state.keys()):
        print("检测到多GPU训练参数，移除'module.'前缀")
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    # 非严格模式加载（兼容部分参数变化）
    load_result = model.load_state_dict(model_state, strict=False)
    print(f"模型参数加载结果: {load_result}")
    return {}


def validate_only(model_path="save_weights/best_model.pth", data_path="../../Dataset/myBreaDM/", device="cuda"):
    # 初始化
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    num_classes = 2
    model = create_model(aux=False, num_classes=num_classes).to(device)

    load_checkpoint(model_path, model)

    # 数据加载
    mean = (0.709)
    std = (0.127)
    val_dataset = DriveDataset(data_path,
                             train=False,
                             transforms=get_transform(train=False, mean=mean, std=std))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        num_workers=8, shuffle=False,
        pin_memory=True, collate_fn=val_dataset.collate_fn
    )

    # 执行验证
    # confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
    # print(f"Validation Results:\n{confmat}")
    # print(f"Dice Coefficient: {dice:.4f}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="/home/wxf/project/Dataset/myBreaDM", help="DRIVE root")
    parser.add_argument("--silent", action="store_true", help="Run in silent mode without generating logs and model files")
    parser.add_argument("--patience", default=10, type=int, help="early stopping patience (default: 10)")
    parser.add_argument("--min-delta", default=1e-4, type=float, help="minimum change in monitored quantity to qualify as an improvement (default: 1e-4)")

    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--val", action="store_true", help="validate_only")
    parser.add_argument("--aux", action="store_true", help="Use auxiliary loss") # fcn增加
    parser.add_argument("--model-type", default="unet", type=str,
                        help="Model type to use: unet, transunet, deeplabv3, fcn, xlstm-unet, tsesnet")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=4.2e-4, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--workers', default=8, type=int,  #  调整数据加载线程
                        help='Number of data loading workers')
    parser.add_argument('--tf32', action='store_true',  #  TF32模式开关
                        help='Enable TF32 computation')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    if args.val == True:
        print("validate_only")
        validate_only()
        exit()

    main(args)


