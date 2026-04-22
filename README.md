# 基于 YOLOv11 的行人和车辆检测

本项目为深圳大学电子与信息工程学院《人工智能》课程实验项目，主要实现基于 **YOLOv11** 的行人和车辆目标检测。项目内容包括数据集下载、环境配置、模型训练、结果推理以及实验报告撰写。

---

## 项目简介 | Project Overview

本实验使用 **Ultralytics YOLOv11n** 预训练模型，在 CPU 环境下对公开数据集进行轻量化微调训练，并实现对图片中的行人和车辆目标检测。

---

## 实验目标 | Objectives

- 学习使用 Anaconda 配置 Python 实验环境
- 掌握通过 Roboflow 下载 YOLO 格式数据集的方法
- 完成 YOLOv11n 的训练、验证与推理流程

---

## 数据集说明 | Dataset

本实验使用 Roboflow 上的公开**行人与车辆检测数据集**，数据集规模约为 **4584 张图像**，包含行人与车辆目标标注，可直接用于 YOLOv11 训练。

#### 数据集目录结构 | Dataset Structure

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## 环境准备 | Environment Setup

本项目使用 Anaconda为实验创建独立环境。

```bash
conda create -n yolo11 python=3.10 -y
conda activate yolo11
pip install ultralytics
pip install roboflow
```

安装完成后可使用以下命令验证：

```bash
python -c "from ultralytics import YOLO; print('安装成功')"
```

---

## 模型训练 | Training

训练脚本核心配置如下：

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='data.yaml',
    epochs=10,
    imgsz=416,
    batch=4,
    device='cpu',
    workers=2,
    project='runs/detect',
    name='yolo11_vp',
    val=True,
)

print("训练完成！权重保存在:", results.save_dir)
```

运行训练：

```bash
python train.py
```

### 训练参数说明 | Training Hyperparameters

- `epochs=10`：训练轮数设置为 10，适合 CPU 环境下的课程实验
- `imgsz=416`：降低输入分辨率以减少训练时间
- `batch=4`：减小 batch size 以适应 CPU 内存与算力限制
- `device='cpu'`：在无 GPU 环境下使用 CPU 训练
- `workers=2`：使用 2 个数据加载线程

---

## 模型推理 | Inference

推理脚本示例：

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo11_vp/weights/best.pt')

results = model.predict(
    source='test1.jpg',
    conf=0.25,
    iou=0.45,
    save=True,
    device='cpu',
)

for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        name = model.names[cls]
        print(f"检测到: {name}, 置信度: {conf:.2f}")
```

运行推理：

```bash
python infer.py
```

检测结果图片默认保存在：

```text
runs/detect/
```

---

## 实验结果 | Results

在 CPU 环境下训练 10 个 epoch 后，实验得到如下结果：

| Epoch | Box(P) | R | mAP@0.5 | mAP@0.5:0.95 |
|------:|-------:|------:|--------:|-------------:|
| 10 | 0.784 | 0.686 | 0.761 | 0.465 |

### 指标说明 | Metric Description

- **Box(P)**：精确率（Precision），表示预测为正样本的目标中有多少是真正正确的  
- **R**：召回率（Recall），表示所有真实目标中有多少被成功检测出来  
- **mAP@0.5**：IoU 阈值为 0.5 时的平均精度均值  
- **mAP@0.5:0.95**：在多个 IoU 阈值下的平均精度均值，评价更严格  

从结果来看，模型已经大致能识别图像中的行人和车辆，具有较高的精确率，但由于模型是在CPU环境下训练且训练轮数较少，模型仍有一定的识别误差，召回率仍有提升空间，说明仍存在一定漏检现象。

---

## 后续改进方向 | Future Work

- 增加训练轮数以进一步提升检测精度
- 尝试使用更大的模型，如 `yolo11s.pt`
- 在 GPU 环境中进行更长时间训练
- 增加更多复杂场景数据以提升模型泛化能力
