import os
import xml.etree.ElementTree as ET

# Update this list exactly to your class names as used in XML files (case-insensitive)
classes = ['apple', 'banana', 'orange']

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmin, xmax, ymin, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return (x_center, y_center, width, height)

def convert_annotation(xml_file, label_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {xml_file}: {e}")
        return False

    size = root.find('size')
    if size is None:
        print(f"Skipping {xml_file}: no size tag found.")
        return False

    w = size.find('width')
    h = size.find('height')
    if w is None or h is None:
        print(f"Skipping {xml_file}: width or height tag missing.")
        return False

    try:
        w = int(w.text)
        h = int(h.text)
    except (TypeError, ValueError):
        print(f"Skipping {xml_file}: invalid width or height values.")
        return False

    if w == 0 or h == 0:
        print(f"Skipping {xml_file}: invalid width or height (w={w}, h={h})")
        return False

    object_count = 0
    with open(label_file, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name')
            if cls is None or cls.text is None:
                print(f"Skipping object in {xml_file} with no class name")
                continue
            cls_name = cls.text.strip().lower()
            if cls_name not in classes:
                print(f"Skipping unknown class '{cls_name}' in {xml_file}")
                continue
            cls_id = classes.index(cls_name)
            xmlbox = obj.find('bndbox')
            if xmlbox is None:
                print(f"Skipping object in {xml_file} with no bounding box")
                continue
            try:
                xmin = float(xmlbox.find('xmin').text)
                ymin = float(xmlbox.find('ymin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymax = float(xmlbox.find('ymax').text)
            except (AttributeError, TypeError, ValueError):
                print(f"Skipping object in {xml_file} due to invalid bounding box coordinates")
                continue

            bbox = convert_bbox((w, h), (xmin, xmax, ymin, ymax))
            out_file.write(f"{cls_id} {' '.join(f'{a:.6f}' for a in bbox)}\n")
            object_count += 1

    return object_count > 0  # True if label file is non-empty, False if empty

def process_folder(images_dir, labels_dir):
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    total_files = 0
    empty_label_files = 0

    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        xml_path = os.path.join(images_dir, xml_filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')

        if not os.path.exists(xml_path):
            print(f"Warning: XML file not found for {filename}")
            continue

        total_files += 1
        has_labels = convert_annotation(xml_path, label_path)
        if not has_labels:
            empty_label_files += 1

    print(f"Processed {total_files} images in {images_dir}")
    print(f"Empty label files count: {empty_label_files}")

    return total_files, empty_label_files

if __name__ == "__main__":
    # Update these paths to your dataset folders
    train_images_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\train"
    train_labels_dir = r"D:\Random Projects\Fruit Images for Object Detection\train_zip\labels\train"
    test_images_dir = r"D:\Random Projects\Fruit Images for Object Detection\test_zip\test"
    test_labels_dir = r"D:\Random Projects\Fruit Images for Object Detection\test_zip\labels\test"

    print("Processing TRAIN set:")
    train_total, train_empty = process_folder(train_images_dir, train_labels_dir)

    print("\nProcessing TEST set:")
    test_total, test_empty = process_folder(test_images_dir, test_labels_dir)

    print("\nSummary:")
    print(f"TRAIN: {train_total} images, {train_empty} empty label files")
    print(f"TEST: {test_total} images, {test_empty} empty label files")
