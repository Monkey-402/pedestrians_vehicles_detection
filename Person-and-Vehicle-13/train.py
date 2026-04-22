from ultralytics import YOLO

 #加载YOLOv11 nano预训练权重（首次运行自动下载，约5MB）
model = YOLO('yolo11n.pt')

 #注意：data填写data.yaml的完整路径
 #如果train.py和数据集文件夹放在同一目录，可以用下面的相对路径：
results = model.train(
 data='data.yaml',
 epochs=10, # CPU训练建议10轮，约30-60分钟
 imgsz=416, #降低分辨率以加快CPU训练速度
 batch=4, # CPU训练batch设小一点
 device='cpu', #无GPU则使用CPU
 workers=2,
 project='runs/detect',
 name='yolo11_vp',
 val=True,
 )

print("训练完成！权重保存于:", results.save_dir)