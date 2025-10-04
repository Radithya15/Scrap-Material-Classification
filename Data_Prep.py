import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time

DATA_DIR = 'TrashNet_dataset' 

IMAGE_SIZE = 224
BATCH_SIZE = 32
TRAIN_RATIO = 0.7 
VAL_RATIO = 0.15 
TEST_RATIO = 0.15 

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

   
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_raw, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) 
    test_loader = DataLoader(test_dataset_raw, batch_size=1, shuffle=False, num_workers=0)

    class_names = full_dataset.classes
    NUM_CLASSES = len(class_names)
    print(f"Classes: {class_names} (Total: {NUM_CLASSES})")
    print("\nStage 1 (Data Prep) completed successfully. Ready for Stage 2.")

   

except FileNotFoundError:
    print("\n\n------------------- FILE NOT FOUND ERROR -------------------")
    print(f"CRITICAL: The script cannot find the folder structure at: {DATA_DIR}")
    print("ACTION: You MUST manually check the path in the 'DATA_DIR' line and correct it.")
    print("----------------------------------------------------------\n")
except RuntimeError as e:
     print(f"\n\nCRITICAL ERROR: {e}")
     print("ACTION: Ensure your data directory has images organized into class subfolders.")