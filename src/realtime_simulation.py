import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import glob
import time


DATA_DIR = 'TrashNet_dataset' 
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.85 # Threshold to flag a low confidence prediction
TORCHSCRIPT_MODEL = 'models/classification_model.pt'
RESULTS_CSV = 'results/simulation_results.csv'

# Ensuring the results directory exists
os.makedirs('results', exist_ok=True) 

# Replicating the test-time preprocessing
inference_transforms = transforms.Compose([
    transforms.Resize(256),                  
    transforms.CenterCrop(IMAGE_SIZE),       
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_inference_model():
    """Loads the TorchScript model."""
    try:
        model = torch.jit.load(TORCHSCRIPT_MODEL)
        model.eval()
        return model
    except Exception as e:
        print(f"FATAL: Could not load TorchScript model at {TORCHSCRIPT_MODEL}. Error: {e}")
        return None

def classify_frame(image_path, model):
    """Takes 1 image/frame and outputs predicted class & confidence."""
    
    # Preprocessing the image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = inference_transforms(image).unsqueeze(0)
    except Exception:
        return "ERROR", 0.0, "Could not process image."

    # Performing Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Getting Prediction and Confidence
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_index = torch.max(probabilities, 0)
    
    predicted_class = CLASS_NAMES[predicted_index.item()]
    confidence_score = confidence.item()
    
    # Checking Confidence Threshold
    flag = "NORMAL"
    if confidence_score < CONFIDENCE_THRESHOLD:
        flag = "LOW_CONFIDENCE_FLAG"
        
    return predicted_class, confidence_score, flag

def simulate_conveyor(model):
    """Builds a dummy conveyor simulation  and logs output."""
    
    all_results = []
    
    
    image_paths = []
    for class_folder in CLASS_NAMES:
        image_paths.extend(glob.glob(os.path.join(DATA_DIR, class_folder, '*.jpg')))
        image_paths.extend(glob.glob(os.path.join(DATA_DIR, class_folder, '*.png')))
    

    if not image_paths:
        print("ERROR: No images found in the dataset directory. Simulation aborted.")
        return
        
    # Simulating processing a fixed number of frames (using 50 for a quick demo)
    FRAMES_TO_PROCESS = min(200, len(image_paths))
    frames_to_process = random.sample(image_paths, FRAMES_TO_PROCESS)
    
    print(f"\n--- Stage 4: Simulated Real-Time Conveyor ---")
    print(f"Processing {FRAMES_TO_PROCESS} frames...")

    for i, image_path in enumerate(frames_to_process):
        
        time.sleep(0.01) 
        
        predicted_class, confidence, flag = classify_frame(image_path, model)
        
        true_class = os.path.basename(os.path.dirname(image_path))
        
        log_message = f"Frame {i+1}: True={true_class:<10} | Pred={predicted_class:<10} | Conf={confidence:.4f} | Status={flag}"
        print(log_message)

        all_results.append({
            'frame_id': i + 1,
            'image_path': image_path,
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_flag': flag
        })

    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    
    print(f"\nSimulation Complete. Results saved to {RESULTS_CSV} (Deliverable: /results/output CSV).")
    print("-" * 50)


if __name__ == '__main__':
    inference_model = load_inference_model()
    
    if inference_model:
        simulate_conveyor(inference_model)