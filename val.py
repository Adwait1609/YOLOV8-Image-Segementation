from ultralytics import YOLO
model = YOLO("C:/Users/USER/OneDrive/Desktop/data/runs/segment/train5/weights/best.pt")

results = model("C:/Users/USER/OneDrive/Desktop/data/test/images/90_Color_png.rf.182daacfe65aeaec05d4db0939ddce87.jpg",save=True)  # predict on an image