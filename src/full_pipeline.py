import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

# Data prep parameters

DATA_DIR = 'TrashNet_dataset' 

IMAGE_SIZE = 224
BATCH_SIZE = 32
TRAIN_RATIO = 0.7 
VAL_RATIO = 0.15 
TEST_RATIO = 0.15 


# Preprocessing & Augmentation

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),       
    transforms.RandomRotation(10),           
    transforms.ToTensor(),                   
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),                  
    transforms.CenterCrop(IMAGE_SIZE),       
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Loading the Dataset & Spliting into train/val/test

print(f"Loading dataset from: {DATA_DIR}")
try:
    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    full_size = len(full_dataset)

    if full_size == 0:
        raise RuntimeError(f"Found 0 images in the directory: {DATA_DIR}. Check your path.")

    train_size = int(TRAIN_RATIO * full_size)
    val_size = int(VAL_RATIO * full_size)
    test_size = full_size - train_size - val_size

    train_dataset, val_dataset_raw, test_dataset_raw = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transforms
    val_dataset_raw.dataset.transform = val_test_transforms
    test_dataset_raw.dataset.transform = val_test_transforms

    print(f"Dataset loaded with {full_size} total images.")
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset_raw)}, Test size: {len(test_dataset_raw)}")

    # Creating DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_raw, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) 
    test_loader = DataLoader(test_dataset_raw, batch_size=1, shuffle=False, num_workers=0) # Batch size 1 for inference

    class_names = full_dataset.classes
    NUM_CLASSES = len(class_names)
    print(f"Classes: {class_names} (Total: {NUM_CLASSES})")
    print("\nStage 1 (Data Prep) completed successfully.")

except Exception as e:
    print(f"\nCRITICAL ERROR in Stage 1: {e}")
    exit()



# Model Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 
MODEL_SAVE_PATH = 'models/best_model_weights.pth'
TORCHSCRIPT_SAVE_PATH = 'models/classification_model.pt'

os.makedirs('models', exist_ok=True) 

# Using ResNet18 with Transfer Learning
def initialize_model(num_classes):
    """Initializes ResNet18, loads pretrained weights, and modifies the final layer."""
    
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freezing all original parameters 
    for param in model_ft.parameters():
        param.requires_grad = False

    # Modifying the final fully connected layer
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    return model_ft.to(DEVICE)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Trains the model and validates after each epoch."""
    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training loop
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)
                total_predictions += labels.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct_predictions.double() / total_predictions
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    end_time = time.time()
    print(f'\nTraining complete in {(end_time - start_time)/60:.2f} minutes. Best Val Acc: {best_accuracy:.4f}')
    
    # Load best model weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    return model

def evaluate_model(model, test_loader, class_names, device):
    """Evaluates the model on the test set and prints detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=list(range(len(class_names))))
    cm = confusion_matrix(all_labels, all_preds)

    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print("\n" + "=" * 50)
    print("FINAL TEST METRICS (Stage 2)")
    print(f"Overall Test Accuracy: {accuracy:.4f}")
    print("\nPer-Class Metrics (Precision/Recall/F1):")
    print(metrics_df.to_markdown(index=False))
    print("\nConfusion Matrix (Rows=True Label, Cols=Predicted Label):")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    # Save confusion matrix as an image
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    print("\nConfusion matrix saved as results/confusion_matrix.png")

    print("=" * 50)

    return accuracy, cm, metrics_df


# Model Conversion
def convert_to_torchscript(model, model_name=TORCHSCRIPT_SAVE_PATH):
    """Converts a trained PyTorch model to TorchScript format for lightweight deployment."""
    
    model.eval()
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    traced_script_module = torch.jit.trace(model, example_input)
    torch.jit.save(traced_script_module, model_name)
    
    print(f"\nStage 3: Model successfully converted and saved as {model_name} (TorchScript)!")
    print("-" * 50)



if __name__ == '__main__':
    model = initialize_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    
    evaluate_model(trained_model, test_loader, class_names, DEVICE)

    convert_to_torchscript(trained_model)

    print("\nStage 2 & 3 completed. Ready for Stage 4: Simulated Real-Time Loop.")