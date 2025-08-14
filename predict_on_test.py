from ultralytics import YOLO
import os

def predict_on_test_images():
    # Load the best trained model weights
    model = YOLO(r'D:/Random Projects/Fruit Images for Object Detection/runs/detect/fruit_yolov8_training/weights/best.pt')

    # Path to your test images
    test_images_path = r'D:/Random Projects/Fruit Images for Object Detection/dataset/images/test'

    # Run inference on the test images
    # The results will be saved in runs/detect/predict/ by default
    results = model.predict(
        source=test_images_path,
        save=True,  # Save predicted images with bounding boxes
        imgsz=640,  # Image size (should match training)
        conf=0.25,  # Confidence threshold for detections
        iou=0.7,    # IoU threshold for Non-Maximum Suppression
        device='cpu' # use CPU for inference; change to 'cuda' if GPU available
    )

    print(f"Predictions saved to: {model.predictor.save_dir}")

if __name__ == "__main__":
    predict_on_test_images()
