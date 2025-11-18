import os
import json
import shutil
from collections import defaultdict
import yaml
from tqdm import tqdm

# List of food-related classes from COCO
FOOD_CLASSES = [
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "cup", "fork", "knife", "spoon", "bowl"
]

def convert_coco_to_yolo(coco_dataset_dir, output_dir):
    """
    Converts a COCO-formatted dataset to YOLO format, filtering for specific food classes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ["train", "val"]:
        print(f"Processing {split} split...")
        
        annotation_file = os.path.join(coco_dataset_dir, "annotations", f"instances_{split}2017.json")
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found: {annotation_file}")
            continue

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Get category mapping
        coco_categories = coco_data['categories']
        food_cat_ids = {cat['id']: cat['name'] for cat in coco_categories if cat['name'] in FOOD_CLASSES}
        
        if not food_cat_ids:
            print("No food classes found in the dataset.")
            return

        yolo_cat_ids = {name: i for i, name in enumerate(food_cat_ids.values())}
        coco_to_yolo_cat_id = {coco_id: yolo_cat_ids[name] for coco_id, name in food_cat_ids.items()}

        # Create directories
        img_out_dir = os.path.join(output_dir, "images", split)
        label_out_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        # Group annotations by image
        annotations_by_image = defaultdict(list)
        for ann in coco_data['annotations']:
            if ann['category_id'] in food_cat_ids:
                annotations_by_image[ann['image_id']].append(ann)

        # Process images
        for img_info in tqdm(coco_data['images'], desc=f"Converting {split} set"):
            img_id = img_info['id']
            if img_id in annotations_by_image:
                # Symlink image
                src_img_path = os.path.join(coco_dataset_dir, "images", f"{split}2017", img_info['file_name'])
                dst_img_path = os.path.join(img_out_dir, img_info['file_name'])
                if not os.path.exists(dst_img_path):
                    os.symlink(os.path.abspath(src_img_path), dst_img_path)

                # Create YOLO label file
                label_path = os.path.join(label_out_dir, os.path.splitext(img_info['file_name'])[0] + ".txt")
                with open(label_path, 'w') as f:
                    for ann in annotations_by_image[img_id]:
                        coco_cat_id = ann['category_id']
                        yolo_cat_id = coco_to_yolo_cat_id[coco_cat_id]
                        
                        # Convert COCO bbox to YOLO format
                        bbox = ann['bbox']
                        x_center = (bbox[0] + bbox[2] / 2) / img_info['width']
                        y_center = (bbox[1] + bbox[3] / 2) / img_info['height']
                        width = bbox[2] / img_info['width']
                        height = bbox[3] / img_info['height']
                        
                        f.write(f"{yolo_cat_id} {x_center} {y_center} {width} {height}\n")

    # Create dataset.yaml
    dataset_yaml_path = os.path.join(output_dir, "dataset.yaml")
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': "images/train",
        'val': "images/val",
        'names': list(yolo_cat_ids.keys())
    }
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)

    print("\nDataset conversion complete!")
    print(f"YOLO-formatted dataset for food classes is at: {os.path.abspath(output_dir)}")
    print(f"Dataset YAML file: {dataset_yaml_path}")

if __name__ == "__main__":
    convert_coco_to_yolo("./coco-dataset", "./food-dataset")
