import os

def remove_unlabeled_images_and_xmls(images_dir, xmls_dir, labels_dir):
    removed_images = 0
    removed_xmls = 0

    # List all images in the images folder
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        img_path = os.path.join(images_dir, img_file)
        xml_file = os.path.splitext(img_file)[0] + '.xml'
        xml_path = os.path.join(xmls_dir, xml_file)

        # Check if label file exists and is not empty
        if not os.path.exists(label_path):
            print(f"Label file missing for {img_file}, removing image and XML.")
            # Remove image
            os.remove(img_path)
            removed_images += 1
            # Remove xml if exists
            if os.path.exists(xml_path):
                os.remove(xml_path)
                removed_xmls += 1
        else:
            # Check if label file is empty
            if os.path.getsize(label_path) == 0:
                print(f"Label file empty for {img_file}, removing image and XML.")
                os.remove(img_path)
                removed_images += 1
                if os.path.exists(xml_path):
                    os.remove(xml_path)
                    removed_xmls += 1

    print(f"Removed {removed_images} images and {removed_xmls} XML files with missing or empty labels.")

if __name__ == "__main__":
    # Update these paths to match your folder structure

    # Training set
    train_images_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\train"
    train_xmls_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\train"  # Assuming XMLs are here. Change if different.
    train_labels_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\labels\train"

    # Test set
    test_images_dir = r"D:\Random Projects\Fruit Images for Object Detection\test_zip\test"
    test_xmls_dir = r"D:\Random Projects\Fruit Images for Object Detection\test_zip\test"  # Assuming XMLs are here. Change if different.
    test_labels_dir = r"D:\Random Projects\Fruit Images for Object Detection\test_zip\labels\test"

    print("Processing TRAIN set:")
    remove_unlabeled_images_and_xmls(train_images_dir, train_xmls_dir, train_labels_dir)

    print("\nProcessing TEST set:")
    remove_unlabeled_images_and_xmls(test_images_dir, test_xmls_dir, test_labels_dir)
