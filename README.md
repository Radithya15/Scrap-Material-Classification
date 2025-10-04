# Scrap Material Classification

## Pipeline Overview & Key Decisions

### 1. Dataset & Preprocessing

  * **Dataset Used:** **TrashNet**. I picked this because it provides a perfect foundation with **6 different material classes** (cardboard, glass, metal, paper, plastic, trash), simulating a real scrap sorting challenge.
  * **Data Prep:** The script handles all the necessary steps: splitting the data into Train/Validation/Test, performing basic cleaning, and using augmentation (flips, rotations) on the training images to make the model more robust.

### 2. Modeling & Training (Stage 2)

  * **Architecture:** I used **ResNet18**, a strong, CNN-based model.
  * **Transfer Learning:** To speed things up and achieve good accuracy quickly, I used **transfer learning**. We leverage the powerful features ResNet18 learned from ImageNet and only retrain the final classification layer for our 6 specific material classes.
  * **Evaluation:** The model's performance was measured using the required metrics: **Accuracy, Precision, Recall, and a Confusion Matrix**

### 3. Lightweight Deployment (Stage 3)

  * **Deployment Choice:** I converted the final trained model to **TorchScript** (`.pt` file)
  * **Why TorchScript?** The goal was **lightweight deployment**. TorchScript serializes the model into a format that runs incredibly fast and can be loaded without needing the full PyTorch Python library overhead, making it ideal for a real-time, high-speed sorting system.

### 4. Real-Time Simulation (Stage 4)

  * The `realtime_simulation.py` script mimics a conveyor belt by processing image "frames" at intervals.
  * For each frame, it uses the lightweight TorchScript model to classify the material and logs the prediction and confidence.
  * It also flags the output if the confidence is below a set threshold, showing awareness of misclassification risks.

-----

## How to Run the Pipeline

Here's a straightforward guide to get the entire pipeline running on your machine.

### **Prerequisites**

Make sure you have Python installed, then grab the necessary libraries:

```bash
# Installs PyTorch, torchvision, numpy, pandas, and scikit-learn
pip install torch torchvision numpy pandas scikit-learn tabulate
```

### **Step 1: Train the Model and Convert it**

This script handles data loading, training, evaluation, and saves the deployable model.

```bash
python src\full_pipeline.py
```

  * *This will download the ResNet18 weights and save the TorchScript model to the `models/` folder.*

### **Step 2: Run the Simulated Conveyor Belt**

This script loads the lightweight TorchScript model and runs the real-time simulation logic.

```bash
python src\realtime_simulation.py
```

  * **Result:** A log of 200 frame predictions will print to your console, and a final log file (`results/simulation_results.csv`) will be saved for inspection.
