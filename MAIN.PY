import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tkinter import Tk, filedialog
from pathlib import Path
from collections import defaultdict

# --- 1. DEFINE THE BLUEPRINT (ARCHITECTURE) ---
def create_model():
    # We start with a standard ResNet50
    model = models.resnet50(weights=None)
    
    # We modify the end to match exactly how we trained it in Colab
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    return model

def run_prediction(image_path, model_path):
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize and Load Weights
    model = create_model()
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Define Image Transformation (Must match Colab val_test_transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and Predict
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)

        label = "Real" if pred.item() == 1 else "Fake"
        confidence = conf.item() * 100
        
        print(f"\n{'='*50}")
        print(f"Image: {Path(image_path).name}")
        print(f"Result: {label.upper()}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def select_image_interactive():
    """Open file dialog to select an image"""
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select an Image for Prediction",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        initialdir=r'E:\PROJECTS\MEDSECURE'
    )
    root.destroy()
    return file_path

def test_accuracy(project_dir, model_path):
    """Test model accuracy on test dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    model = create_model()
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Test dataset path
    test_dir = os.path.join(project_dir, 'archive (1)', 'dataset', 'test')
    
    if not os.path.exists(test_dir):
        print(f"Error: Test dataset not found at {test_dir}")
        return
    
    # Get image files
    fake_dir = os.path.join(test_dir, 'Fake')
    real_dir = os.path.join(test_dir, 'Real')
    
    predictions = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    total = 0
    correct = 0
    
    # Valid image extensions
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    
    print("\n" + "="*60)
    print("TESTING MODEL ACCURACY")
    print("="*60)
    
    # Test Fake images (label = 0)
    print("\nTesting FAKE images...")
    fake_count = 0
    if os.path.exists(fake_dir):
        for img_file in os.listdir(fake_dir):
            if os.path.splitext(img_file)[1] in valid_ext:
                try:
                    img_path = os.path.join(fake_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    input_tensor = transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, pred = torch.max(outputs, 1)
                    
                    pred_label = pred.item()
                    true_label = 0  # Fake
                    
                    if pred_label == true_label:
                        predictions['TN'] += 1
                        correct += 1
                    else:
                        predictions['FP'] += 1
                    
                    total += 1
                    fake_count += 1
                except Exception as e:
                    print(f"  Error processing {img_file}: {str(e)}")
    
    print(f"  Processed {fake_count} fake images ✓")
    
    # Test Real images (label = 1)
    print("\nTesting REAL images...")
    real_count = 0
    if os.path.exists(real_dir):
        for img_file in os.listdir(real_dir):
            if os.path.splitext(img_file)[1] in valid_ext:
                try:
                    img_path = os.path.join(real_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    input_tensor = transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, pred = torch.max(outputs, 1)
                    
                    pred_label = pred.item()
                    true_label = 1  # Real
                    
                    if pred_label == true_label:
                        predictions['TP'] += 1
                        correct += 1
                    else:
                        predictions['FN'] += 1
                    
                    total += 1
                    real_count += 1
                except Exception as e:
                    print(f"  Error processing {img_file}: {str(e)}")
    
    print(f"  Processed {real_count} real images ✓")
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    precision = (predictions['TP'] / (predictions['TP'] + predictions['FP']) * 100) if (predictions['TP'] + predictions['FP']) > 0 else 0
    recall = (predictions['TP'] / (predictions['TP'] + predictions['FN']) * 100) if (predictions['TP'] + predictions['FN']) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total Images Tested: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {predictions['TP']}")
    print(f"  True Negatives (TN):  {predictions['TN']}")
    print(f"  False Positives (FP): {predictions['FP']}")
    print(f"  False Negatives (FN): {predictions['FN']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1-Score:  {f1:.2f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    PROJECT_DIR = r'E:\PROJECTS\MEDSECURE'
    MODEL_PATH = os.path.join(PROJECT_DIR, 'best_model.pth')
    
    # Check for test accuracy command
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['test', 'accuracy', '--test']:
        test_accuracy(PROJECT_DIR, MODEL_PATH)
    # Check if image path provided as command-line argument
    elif len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
        print(f"Using image: {IMAGE_PATH}")
        run_prediction(IMAGE_PATH, MODEL_PATH)
    else:
        # Open interactive file dialog
        print("Opening file browser...")
        IMAGE_PATH = select_image_interactive()
        
        if not IMAGE_PATH:
            print("No image selected. Exiting.")
            sys.exit(0)
        
        run_prediction(IMAGE_PATH, MODEL_PATH)