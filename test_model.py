from ultralytics import YOLO

def test_yolov8_model():
    # Load the best trained model weights
    model = YOLO(r'D:/Random Projects/Fruit Images for Object Detection/runs/detect/fruit_yolov8_training/weights/best.pt')

    # Evaluate the model on the test set
    results = model.val(
        data=r'D:/Random Projects/Fruit Images for Object Detection/data.yaml',  # path to your data.yaml
        split='test',                  # Evaluate on the test set
        imgsz=640,                   # image size (should match training)
        batch=4,                     # batch size (can be adjusted for testing)
        device='cpu',                # use CPU for evaluation; change to 'cuda' if GPU available
        name='fruit_yolov8_test'    # custom name for the test run (output folder)
    )

    # Print evaluation results
    print(results)

if __name__ == "__main__":
    test_yolov8_model()
