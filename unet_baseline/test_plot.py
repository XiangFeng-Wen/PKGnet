from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from scipy.optimize import curve_fit
from PIL import Image

# Tofts模型定义
def tofts_model(t, K_trans, V_p, V_e, baseline):
    """
    Tofts模型公式：描述DCE-MRI中的药代动力学特性。
    
    参数：
    t -- 时间序列
    K_trans -- 转运速率常数
    V_p -- 血容量分数
    V_e -- 细胞外间隙分数
    baseline -- 基线信号
    """
    # K_trans: 转运速率常数
    # V_p: 血容量分数
    # V_e: 细胞外间隙分数
    # baseline: 基线信号
    return baseline + (K_trans / V_p) * np.cumsum(np.exp(-K_trans / V_e * (t - t[0]))) + V_p * np.array([baseline] * len(t))

# 数据集类，用于加载DCE-MRI图像数据并提取PK参数
class DCE_MRI_PK_Extractor:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*', '*', '*.png'))
    
    def load_image(self, path):
        """
        加载图像并进行预处理。
        """
        image = Image.open(path).convert("L")  # 转为灰度图
        image = np.array(image)  # 转为NumPy数组
        if self.transform:
            image = self.transform(image)  # 可添加数据增强或归一化
        return image

    def extract_pk_parameters(self, image_sequence, time_points):
        """
        从DCE-MRI图像序列中提取PK参数。
        
        参数：
        image_sequence -- 图像序列（每个图像为一个时间点的信号强度）
        time_points -- 时间点数组，表示每帧图像对应的时间
        
        返回：
        PK参数：转运速率常数K_trans，血容量分数V_p，细胞外间隙分数V_e，基线信号
        """
        # 初始化PK参数
        K_trans_init = 0.1
        V_p_init = 0.3
        V_e_init = 0.5
        baseline_init = np.min(image_sequence)

        # 使用Tofts模型进行拟合
        popt, _ = curve_fit(tofts_model, time_points, image_sequence, 
                            p0=[K_trans_init, V_p_init, V_e_init, baseline_init])
        K_trans, V_p, V_e, baseline = popt
        return K_trans, V_p, V_e, baseline

    def process_patient(self, patient_id):
        """
        处理一个患者的所有图像序列并提取PK参数。
        
        参数：
        patient_id -- 患者ID（例如：'SUB1'，'SUB2'等）
        
        返回：
        所有PK参数的字典：{图像序号: (K_trans, V_p, V_e, baseline)}
        """
        patient_images = [path for path in self.image_paths if patient_id in path]
        time_points = np.array([i for i in range(len(patient_images))])
        pk_params = {}

        # 遍历该患者的所有图像并进行PK参数提取
        for idx, image_path in enumerate(patient_images):
            image = self.load_image(image_path)
            K_trans, V_p, V_e, baseline = self.extract_pk_parameters(image, time_points)
            pk_params[image_path] = (K_trans, V_p, V_e, baseline)
        
        return pk_params

# 数据集路径
root_dir = '/home/wxf/project/Dataset/BreaDM'

# 创建PK提取器对象
extractor = DCE_MRI_PK_Extractor(root_dir=root_dir)

# 提取某个患者的PK参数
patient_id = 'SUB1'  # 你可以选择其他患者
pk_params = extractor.process_patient(patient_id)

# 输出某个图像的PK参数
for image_path, (K_trans, V_p, V_e, baseline) in pk_params.items():
    print(f"Image: {image_path}")
    print(f"K_trans: {K_trans:.3f}, V_p: {V_p:.3f}, V_e: {V_e:.3f}, Baseline: {baseline:.3f}")
    
    # 可视化拟合结果
    time_points = np.array([i for i in range(len(pk_params))])
    image_sequence = np.array([extractor.load_image(image_path) for image_path in pk_params.keys()])
    plt.plot(time_points, image_sequence, label='Observed')
    plt.plot(time_points, tofts_model(time_points, K_trans, V_p, V_e, baseline), label='Fitted', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.title('PK Signal Intensity')
    plt.savefig("pk_output.png")


