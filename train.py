from ultralytics import YOLO
import multiprocessing

def train_model():
    # Load a pretrained model (recommended for training)
    model = YOLO("yolov8n-seg.pt")
    
    # Train the model
    results = model.train(data="C:/Users/USER/OneDrive/Desktop/data/data.yaml", epochs=100, imgsz=640)
    return results

def main():
    # Call the function to train the model
    train_results = train_model()
    # You can add more code here if needed, e.g., to process or save `train_results`
    print("Training completed")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Only needed if you are freezing your script
    main()
    