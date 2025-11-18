from ultralytics import YOLO
import os
from PIL import Image
import torch
import subprocess
import json
import yaml
import argparse
from download_coco import download_coco2017
from prepare_food_dataset import convert_coco_to_yolo

def get_device():
    """모델의 정확도 평가 함수"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def prepare_food_dataset():
    """
    Prepares the food dataset for YOLO training.
    Downloads COCO if needed, and then converts it to a food-only dataset.
    """
    coco_dataset_dir = "./coco-dataset"
    food_dataset_dir = "./food-dataset"
    yolo_yaml_path = os.path.join(food_dataset_dir, "dataset.yaml")

    # Step 1: Ensure COCO dataset is available
    coco_annotations_dir = os.path.join(coco_dataset_dir, "annotations")
    if not os.path.exists(coco_annotations_dir) or not os.listdir(coco_annotations_dir):
        print(f"COCO dataset not found or incomplete in '{coco_dataset_dir}'.")
        print("Downloading the COCO dataset...")
        download_coco2017(coco_dataset_dir)

    # Step 2: Check if food dataset is already prepared
    if not os.path.exists(yolo_yaml_path):
        print(f"Food dataset not found in '{food_dataset_dir}'.")
        print("Converting COCO to YOLO food dataset...")
        convert_coco_to_yolo(coco_dataset_dir, food_dataset_dir)
    else:
        print("Food dataset already prepared.")

    return yolo_yaml_path

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train for")
    args = parser.parse_args()

    print("Starting YOLO training script...")

    # --- YOLO Dataset ---
    print("\n--- Dataset Loading (YOLO) ---")
    yolo_yaml_path = prepare_food_dataset()

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

    # Find the latest training run directory
    base_run_dir = 'runs/detect'
    run_name_prefix = 'train_food'
    latest_run_dir = None
    if os.path.exists(base_run_dir):
        run_dirs = sorted([d for d in os.listdir(base_run_dir) if d.startswith(run_name_prefix) and os.path.isdir(os.path.join(base_run_dir, d))])
        if run_dirs:
            latest_run_dir = os.path.join(base_run_dir, run_dirs[-1])
            print(f"Found latest training run at: {latest_run_dir}")

    last_checkpoint_path = None
    if latest_run_dir:
        potential_checkpoint = os.path.join(latest_run_dir, 'weights/last.pt')
        if os.path.exists(potential_checkpoint):
            last_checkpoint_path = potential_checkpoint

    model_to_train = None
    resume_training = False
    completed_epochs = 0

    if last_checkpoint_path:
        print(f"Found checkpoint at {last_checkpoint_path}. Resuming training.")
        # When resuming, YOLO automatically loads the state from the checkpoint
        model_to_train = YOLO(last_checkpoint_path)
        resume_training = True
        try:
            # Load checkpoint separately to read metadata
            ckpt = torch.load(last_checkpoint_path, map_location=get_device())
            completed_epochs = ckpt.get('epoch', -1) + 1  # epoch is 0-indexed
            print(f"Checkpoint is from epoch {completed_epochs}.")
        except Exception as e:
            print(f"Could not read epoch from checkpoint: {e}. Assuming 0 completed epochs.")
            completed_epochs = 0

    elif os.path.exists(YOLO_MODEL_PATH):
        print(f"Loading saved YOLO model from {YOLO_MODEL_PATH}")
        model_to_train = YOLO(YOLO_MODEL_PATH)
    else:
        print("No saved YOLO model found. Starting with pre-trained 'yolov8n.pt'.")
        model_to_train = YOLO('yolov8n.pt')
    
    print("YOLO model initialized.")

    # --- Configuration ---
    EPOCHS = args.epochs
    if EPOCHS is None:
        try:
            if resume_training:
                prompt = f"Model has been trained for {completed_epochs} epochs. Enter the new TOTAL number of epochs to train for: "
            else:
                prompt = "Enter the number of epochs to train for: "
            num_epochs_input = input(prompt)
            EPOCHS = int(num_epochs_input)
        except (ValueError, EOFError):
            print(f"\nInvalid input. Exiting.")
            return

    if resume_training and EPOCHS <= completed_epochs:
        print(f"\nError: The model has already been trained for {completed_epochs} epochs.")
        print(f"Please provide a total number of epochs greater than {completed_epochs} to continue training.")
        return

    BATCH_SIZE = 8

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
    
    # Look for images in the validation directory of the prepared dataset
    val_images_dir = os.path.join(os.path.dirname(yolo_yaml_path), "images", "val")
    
    if os.path.exists(val_images_dir) and os.listdir(val_images_dir):
        # Get the first image file from the validation set
        test_image_name = next((f for f in os.listdir(val_images_dir) if os.path.isfile(os.path.join(val_images_dir, f))), None)
        
        if test_image_name:
            image_path_detection = os.path.join(val_images_dir, test_image_name)
            print(f"Performing inference on image: {image_path_detection}")

            yolo_results = model_yolo_trained(image_path_detection)

            # Process and save the result for the single image
            if yolo_results:
                r = yolo_results[0]  # Get the first result
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                output_image_path = "yolo_inference_result.jpg"
                im.save(output_image_path)
                print(f"Detections saved to {output_image_path}")
        else:
            print("No image files found in the validation directory for inference example.")
    else:
        print("No images found for inference example.")
        
    print("YOLO inference example finished.")
    print("\nYOLO script finished.")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
