from ultralytics import YOLO

def train_yolov8_model():
    # Load the YOLOv8n pretrained model (nano, lightweight, good for CPU training)
    model = YOLO('yolov8n.pt')

    # Start training
    results = model.train(
        data=r'D:/Random Projects/Fruit Images for Object Detection/data.yaml',  # path to your data.yaml
        epochs=50,                   # number of epochs to train
        imgsz=640,                   # image size (resize input images)
        batch=4,                     # batch size, reduce if memory issues on CPU
        device='cpu',                # force CPU training; remove or change to 'cuda' if GPU available
        name='fruit_yolov8_training' # custom name for the training run (output folder)
    )

    # Optional: print final training metrics or results summary
    print(results)

if __name__ == "__main__":
    train_yolov8_model()
