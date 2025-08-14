import os
import random
import shutil

def split_data(source_images_dir, source_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.2):
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    num_val_samples = int(len(image_files) * split_ratio)
    val_files = image_files[:num_val_samples]

    print(f"Total images in {source_images_dir}: {len(image_files)}")
    print(f"Moving {len(val_files)} images and labels to validation set...")

    for img_file in val_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'

        # Move image
        shutil.move(os.path.join(source_images_dir, img_file), os.path.join(val_images_dir, img_file))
        # Move label
        if os.path.exists(os.path.join(source_labels_dir, label_file)):
            shutil.move(os.path.join(source_labels_dir, label_file), os.path.join(val_labels_dir, label_file))
        else:
            print(f"Warning: Label file {label_file} not found for image {img_file}")

    print("Data splitting complete.")

if __name__ == "__main__":
    train_images_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\train"
    train_labels_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\labels\train"
    val_images_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\val"
    val_labels_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\labels\val"

    split_data(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.2)
