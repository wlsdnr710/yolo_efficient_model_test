from ultralytics import YOLO
import os
from PIL import Image
import torch
import subprocess
import json
import yaml
import argparse

def get_device():
    """Detects and returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def prepare_coco_dataset():
    """
    Prepares the dataset.yaml file for the local COCO dataset.
    """
    dataset_dir = "./coco-dataset"
    yolo_yaml_path = os.path.join(dataset_dir, "dataset.yaml")

    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' is missing or empty.")
        print("Please download the COCO dataset and place it in the correct directory.")
        return None

    # Assuming a standard COCO structure
    train_images = 'images/train2017'
    val_images = 'images/val2017'
    annotations_file = 'annotations/instances_train2017.json'

    if not os.path.exists(os.path.join(dataset_dir, train_images)) or \
       not os.path.exists(os.path.join(dataset_dir, val_images)) or \
       not os.path.exists(os.path.join(dataset_dir, annotations_file)):
        print("Error: The coco-dataset directory does not have the expected structure.")
        print("Expected structure: images/train2017, images/val2017, annotations/instances_train2017.json")
        return None

    with open(os.path.join(dataset_dir, annotations_file), 'r') as f:
        coco_data = json.load(f)
    
    class_names = [cat['name'] for cat in coco_data['categories']]
    
    dataset_yaml = {
        'path': os.path.abspath(dataset_dir),
        'train': train_images,
        'val': val_images,
        'names': class_names
    }
    
    with open(yolo_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)
        
    print("Dataset ready.")
    return yolo_yaml_path

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train for")
    args = parser.parse_args()

    print("Starting YOLO training script...")

    # --- Configuration ---
    EPOCHS = args.epochs
    if EPOCHS is None:
        try:
            num_epochs_input = input("Enter the number of epochs to train for: ")
            EPOCHS = int(num_epochs_input)
        except ValueError:
            print("Invalid input. Using default number of epochs: 2")
            EPOCHS = 2
    BATCH_SIZE = 4
    
    # --- YOLO Dataset ---
    print("\n--- Dataset Loading (YOLO) ---")
    yolo_yaml_path = prepare_coco_dataset()

    if not yolo_yaml_path:
        print("Failed to get dataset. Exiting.")
        return

    # --- Define and Load YOLO Model ---
    print("\n--- Model Initialization (YOLO) ---")
    MODEL_DIR = "./trained_models"
    YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolo_food_detector.pt")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model directory: {MODEL_DIR}")

    train_run_dir = 'runs/detect/train_food'
    last_checkpoint_path = os.path.join(train_run_dir, 'weights/last.pt')
    
    model_to_train = None
    resume_training = False
    if os.path.exists(last_checkpoint_path):
        print(f"Found checkpoint at {last_checkpoint_path}. Resuming training.")
        model_to_train = YOLO(last_checkpoint_path)
        resume_training = True
    elif os.path.exists(YOLO_MODEL_PATH):
        print(f"Loading saved YOLO model from {YOLO_MODEL_PATH}")
        model_to_train = YOLO(YOLO_MODEL_PATH)
    else:
        print("No saved YOLO model found. Starting with pre-trained 'yolov8n.pt'.")
        model_to_train = YOLO('yolov8n.pt')
    
    print("YOLO model initialized.")

    # --- Train YOLO ---
    print("\n--- Starting YOLO Training ---")
    device = get_device()
    print(f"Using device: {device}")

    results = model_to_train.train(
        data=yolo_yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=640,
        project='runs/detect',
        name='train_food',
        device=device,
        save_period=1,
        resume=resume_training
    )
    print("YOLO training finished.")

    # --- Save YOLO Model ---
    print("\n--- Saving YOLO Model ---")
    model_to_train.save(YOLO_MODEL_PATH)
    print(f"Final YOLO model explicitly saved to {YOLO_MODEL_PATH}")
    
    best_yolo_model_path = os.path.join(results.save_dir, 'weights/best.pt')
    if not os.path.exists(best_yolo_model_path):
        best_yolo_model_path = YOLO_MODEL_PATH

    # --- Inference Example ---
    print("\n--- Starting YOLO Inference Example ---")
    model_yolo_trained = YOLO(best_yolo_model_path)
    
    images_dir = os.path.join(os.path.dirname(yolo_yaml_path), "images")
    if os.path.exists(images_dir) and os.listdir(images_dir):
        test_image_name = os.listdir(images_dir)[0]
        image_path_detection = os.path.join(images_dir, test_image_name)
        print(f"Performing inference on image: {image_path_detection}")

        yolo_results = model_yolo_trained(image_path_detection)

        for r in yolo_results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            output_image_path = "yolo_inference_result.jpg"
            im.save(output_image_path)
            print(f"Detections saved to {output_image_path}")
    else:
        print("No images found for inference example.")
        
    print("YOLO inference example finished.")
    print("\nYOLO script finished.")

if __name__ == '__main__':
    main()
