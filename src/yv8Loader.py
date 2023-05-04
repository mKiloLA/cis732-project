import ultralytics
from ultralytics import YOLO
import wandb
from wandb_callback import callbacks

ultralytics.checks()
  
wandb.login(key='fc8f2dbcf13fb574caac96ea0f5db245d875c25d')
wandb.init(project="cis732-project",settings=wandb.Settings(start_method="thread"))

# ymodel = YOLO("/homes/zcoster/CIS732_Project/ultralytics/yolov8x.pt")
# ymodel = YOLO("/homes/zcoster/runs/detect/train6/weights/best.pt")
ymodel = YOLO("/homes/zcoster/runs/detect/baseline/COCOptBaseline.pt")

for event,func in callbacks.items():
    ymodel.add_callback(event,func)

# ymodel.val(data="/homes/zcoster/CIS732_Project/test/testData.yaml")
ymodel.val(data="/homes/zcoster/CIS732_Project/BaleDataset/BaleDataset.yaml", split="test")
# ymodel.train(data="/homes/zcoster/CIS732_Project/BaleDataset/BaleDataset.yaml", epochs=200, imgsz=640)
    
