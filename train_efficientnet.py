import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
from torchvision import transforms
from PIL import Image
import os
import random
import argparse

# It is recommended to install timm for this script to work
# pip install timm

def get_device():
    """Detects and returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def calculate_accuracy(model, data_loader, device):
    """Calculates the accuracy of the model on a given dataset."""
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    model.train()  # Set the model back to training mode
    return accuracy

import argparse

def main():
    parser = argparse.ArgumentParser(description="EfficientNet Training Script")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs to train for")
    args = parser.parse_args()

    print("Starting EfficientNet training script...")

    # --- Configuration ---
    BATCH_SIZE = 4
    NUM_EPOCHS = args.num_epochs

    if NUM_EPOCHS is None:
        try:
            num_epochs_input = input("Enter the number of epochs to train for: ")
            NUM_EPOCHS = int(num_epochs_input)
        except ValueError:
            print("Invalid input. Using default number of epochs: 2")
            NUM_EPOCHS = 2
    
    # --- Define and Load EfficientNet Model ---
    print("\n--- Model Initialization (EfficientNet) ---")
    MODEL_DIR = "./trained_models"
    EFFICIENTNET_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_food_classifier.pth")
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "efficientnet_checkpoint.pth")
    HF_MODEL_NAME = 'timm/efficientnet_b0.ra_in1k'

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model directory: {MODEL_DIR}")

    device = get_device()
    print(f"Using device: {device}")

    num_classes = 101
    print(f"Creating EfficientNet model '{HF_MODEL_NAME}' with {num_classes} output classes...")
    model_efficientnet = timm.create_model(HF_MODEL_NAME, pretrained=True, num_classes=num_classes)
    
    # --- EfficientNet Dataset (Food-101) ---
    print("\n--- Dataset Loading (EfficientNet) ---")
    dataset_dir_classification = "./food-101-dataset"
    
    try:
        # Check if the dataset exists. If not, download it.
        print("Checking for Food-101 dataset...")
        dataset_exists = os.path.exists(dataset_dir_classification) and len(os.listdir(dataset_dir_classification)) > 0
        
        # Image transformations
        data_config = timm.data.resolve_model_data_config(model_efficientnet)
        transform = timm.data.create_transform(**data_config)
        print("Image transformations created.")

        print("Loading Food-101 dataset using torchvision...")
        full_dataset = Food101(
            root=dataset_dir_classification, 
            split="train", 
            transform=transform, 
            download=not dataset_exists
        )
        
        # Using a smaller subset for demonstration purposes
        num_samples_to_use = 500
        subset_indices = random.sample(range(len(full_dataset)), num_samples_to_use)
        train_dataset = Subset(full_dataset, subset_indices)
        
        print(f"Food-101 dataset loaded. Using a subset of {len(train_dataset)} samples for this demo.")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have a working internet connection for download or that the dataset is correctly placed.")
        return

    model_efficientnet = model_efficientnet.to(device)

    optimizer = optim.Adam(model_efficientnet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    # Load from checkpoint or final model
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model_efficientnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    elif os.path.exists(EFFICIENTNET_MODEL_PATH):
        print(f"Loading fully trained model from {EFFICIENTNET_MODEL_PATH} for further training.")
        model_efficientnet.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=device))
        print("Model loaded. Ready for additional training.")
    else:
        print(f"No checkpoint or saved model found. Starting training from scratch.")

    print("Model, optimizer, and criterion initialized.")

    # --- Train EfficientNet ---
    print("\n--- Training Setup (EfficientNet) ---")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    print(f"DataLoader created with batch size {train_loader.batch_size} for {len(train_loader)} batches.")

    # Calculate initial accuracy
    initial_accuracy = calculate_accuracy(model_efficientnet, train_loader, device)
    print(f"Initial Training Accuracy: {initial_accuracy:.2f}%")

    print("\n--- Starting EfficientNet Training ---")
    model_efficientnet.train()

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model_efficientnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Improved progress printing
            progress = (i + 1) / len(train_loader)
            progress_bar = "#" * int(20 * progress) + "-" * (20 - int(20 * progress))
            print(f"\rEpoch [{epoch+1}/{start_epoch + NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}] [{progress_bar}] {progress*100:.1f}%, Loss: {loss.item():.4f}", end="")

        # Print final loss for the epoch on a new line
        print(f"\nEpoch [{epoch+1}/{start_epoch + NUM_EPOCHS}] finished. Average Loss: {running_loss / len(train_loader):.4f}")
        
        # Calculate and print accuracy at the end of each epoch
        accuracy = calculate_accuracy(model_efficientnet, train_loader, device)
        print(f"Training Accuracy: {accuracy:.2f}%")

        # --- Save Checkpoint ---
        print(f"--- Saving checkpoint for epoch {epoch+1} ---")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_efficientnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    print("EfficientNet training finished.")

    # --- Save EfficientNet Model ---
    print("\n--- Saving EfficientNet Model ---")
    torch.save(model_efficientnet.state_dict(), EFFICIENTNET_MODEL_PATH)
    print(f"EfficientNet model saved to {EFFICIENTNET_MODEL_PATH}")

    # --- Inference Example ---
    print("\n--- Starting EfficientNet Inference Example ---")
    # Get a test image and label from the full dataset
    test_indices = list(set(range(len(full_dataset))) - set(subset_indices))
    if not test_indices:
        test_indices = subset_indices # Fallback if all samples were used for training
        
    test_sample_idx = random.choice(test_indices)
    image, label_idx = full_dataset[test_sample_idx]
    true_label = full_dataset.classes[label_idx]
    
    # The image is already a tensor, so we just add the batch dimension
    input_tensor = image.unsqueeze(0).to(device)

    model_efficientnet.eval()
    with torch.no_grad():
        output = model_efficientnet(input_tensor)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = full_dataset.classes[predicted_idx.item()]

    print(f"Inference on a random test image.")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")
    print("EfficientNet inference example finished.")

    print("\nEfficientNet script finished.")

if __name__ == '__main__':
    main()
