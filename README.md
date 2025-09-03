胃癌病理图像分割与分类AI代码运行说明
________________________________________
一、运行环境说明
代码模块基于 Google Colab 环境运行，依赖 GPU 加速。
推荐环境配置如下：
项目	版本 / 说明
Python	3.9+
PyTorch	1.13.1 + CUDA 11.6
torchvision	0.14.1
segmentation-models-pytorch	0.3.3
numpy	1.23.5
pandas	1.5.3
Pillow (PIL)	9.3.0
matplotlib	3.6.3
tqdm	4.64.1
安装依赖：
pip install torch torchvision segmentation-models-pytorch numpy pandas pillow matplotlib tqdm
________________________________________
二、数据准备
1. 数据路径
数据存放于 Google Drive，路径如下：
•	原始图像：/content/drive/MyDrive/Trivial Files/train_org_image_100
•	分割 Mask：/content/drive/MyDrive/Trivial Files/train_mask_100
•	标签文件：/content/drive/MyDrive/Trivial Files/train_label.csv
•	测试图像：/content/drive/MyDrive/Trivial Files/test_images

2. 标签文件格式
train_label.csv 文件需包含两列：
•	image_name：图像文件名（需与原图和 mask 文件名一致）
•	label：类别标签（整数编码，例如 0, 1, 2）

3. 数据对齐要求
•	原图与 mask 必须一一对应，文件名相同（如 img_001.png 与 img_001.png）。
•	图像格式需统一（.png）。
•	mask 为二值图（0=背景，1=癌变区域）。
________________________________________
三、模型与训练配置
1. U-Net 模型（分割）
•	Backbone：ResNet34（ImageNet 预训练权重）
•	输入通道：3（RGB 图像）
•	输出通道：1（二值分割）
•	损失函数：BCEWithLogitsLoss
•	优化器：Adam（lr=1e-4）
•	Batch Size：4 （可以根据实际情况调整）
•	Epoch 数：5（可以根据实际情况调整）
•	评估指标：Dice Score

2. ResNet 模型（分类）
•	基础结构：ResNet18（ImageNet 预训练权重）
•	输入修改：
o	将输入层从 3 通道改为 4 通道（RGB + mask）
•	输出修改：
o	全连接层修改为 Linear(in_features, num_classes=3)
•	损失函数：CrossEntropyLoss
•	优化器：Adam（lr=1e-4）
•	Batch Size：4（可以根据实际情况调整）
•	Epoch 数：5（可以根据实际情况调整）
•	评估指标：分类准确率（Accuracy）
________________________________________
四、运行步骤
1. U-Net 训练模块：gastric_cancer_u_net_train.py
•	输出：
o	训练 Loss 曲线
o	Dice Score 曲线
o	模型权重 U_Net.pth（保存至 /content/drive/MyDrive/AI_Models/）

2. ResNet 训练模块：gastric_cancer_resnet_train.py
•	输出：
o	训练 Loss 曲线
o	准确率曲线
o	模型权重 ResNet.pth（保存至 /content/drive/MyDrive/AI_Models/）

3. 主程序推理：gastric_cancer_main.py
•	加载模型权重：
o	/content/drive/MyDrive/AI_Models/U_Net.pth
o	/content/drive/MyDrive/AI_Models/ResNet.pth

•	运行流程：
1.	U-Net 对输入图像生成预测 mask
2.	将原图与 mask 拼接为 4 通道输入
3.	ResNet 分类输出类别

示例输出（控制台）：
Predicted Class: CancerType1 (1)
Normal: 0.1234
CancerType1: 0.7421
CancerType2: 0.1345

可视化结果（Matplotlib）：
•	左：原始图像
•	中：U-Net 预测 mask
•	右：带预测标签的原图
________________________________________
六、Colab → 本地运行迁移指南
由于代码最初在 Google Colab 上开发，本地 GPU 运行时需注意以下修改：
1.	删除 Google Drive 挂载代码：
from google.colab import drive
drive.mount('/content/drive')

本地运行请删除上述代码，手动修改路径为本地目录，例如：
•	image_dir = "/home/user/data/train_org_image_100"
•	mask_dir = "/home/user/data/train_mask_100"
•	excel_file = "/home/user/data/train_label.csv"

2.	替换感叹号命令 !
o	Colab：
!pip install segmentation-models-pytorch --quiet
!cp U_Net.pth /content/drive/MyDrive/AI_Models/

o	本地：
pip install segmentation-models-pytorch
cp U_Net.pth ./models/

🔹 模型权重保存与加载说明
在代码中，U-Net 和 ResNet 的训练脚本都会在最后保存模型权重，并在推理脚本 (gastric_cancer_main.py) 中加载。由于路径在 Google Colab 和 本地 GPU 环境下不同，需要特别说明。
________________________________________
Google Colab 默认行为
•	保存权重
torch.save(trained_unet.state_dict(), "U_Net.pth")
!cp U_Net.pth /content/drive/MyDrive/AI_Models/

•	最终文件会出现在：
/content/drive/MyDrive/AI_Models/U_Net.pth
/content/drive/MyDrive/AI_Models/ResNet.pth

•	加载权重（在 gastric_cancer_main.py 中）：
U_Net.load_state_dict(torch.load("/content/drive/MyDrive/AI_Models/U_Net.pth"))
ResNet.load_state_dict(torch.load("/content/drive/MyDrive/AI_Models/ResNet.pth"))
________________________________________
本地 GPU 环境推荐做法
在本地，不再使用 drive.mount 和 /content/drive/... 这种路径，推荐建立一个统一的 ./models/ 文件夹 来保存权重。

保存权重（修改训练脚本）：
import os
os.makedirs("./models", exist_ok=True)

torch.save(trained_unet.state_dict(), "./models/U_Net.pth")
torch.save(trained_resnet.state_dict(), "./models/ResNet.pth")
print("Models saved in ./models/")

加载权重（修改推理脚本）：
U_Net.load_state_dict(torch.load("./models/U_Net.pth", map_location=device))
ResNet.load_state_dict(torch.load("./models/ResNet.pth", map_location=device))
