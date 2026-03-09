import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# --- MODEL ARCHITECTURE ---
def create_model():
    """Define the ResNet50 architecture"""
    model = models.resnet50(weights=None)
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

# --- INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="MedSecure - Medicine Detection API",
    description="Counterfeit Medicine Detection using ResNet50",
    version="1.0.0"
)

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL VARIABLES ---
PROJECT_DIR = r'E:\PROJECTS\MEDSECURE'
MODEL_PATH = os.path.join(PROJECT_DIR, 'best_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model once on startup
MODEL = None
TRANSFORM = None

def load_model():
    """Load model and transforms"""
    global MODEL, TRANSFORM
    
    if MODEL is None:
        MODEL = create_model()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()
    
    if TRANSFORM is None:
        TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    return MODEL, TRANSFORM

# Load model at startup
load_model()

# --- ENDPOINTS ---

@app.get("/", tags=["Status"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "MedSecure Medicine Detection API",
        "version": "1.0.0",
        "endpoints": {
            "info": "GET /model-info - Get model information",
            "health": "GET /health - Check API health",
            "predict": "POST /predict - Upload image for prediction",
            "test": "GET /test - Run accuracy test on test dataset",
            "docs": "GET /docs - Interactive API documentation"
        }
    }

@app.get("/health", tags=["Status"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH
    }

@app.get("/model-info", tags=["Status"])
async def model_info():
    """Get model information"""
    return {
        "model": "ResNet50",
        "classes": ["Fake", "Real"],
        "accuracy": "98.44%",
        "precision": "99.65%",
        "recall": "97.91%",
        "f1_score": "98.77",
        "input_size": [224, 224],
        "framework": "PyTorch",
        "device": str(DEVICE)
    }

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict if a medicine is Real or Fake
    
    - **file**: Image file to analyze (jpg, png, jpeg, bmp)
    
    Returns:
    - **prediction**: "Real" or "Fake"
    - **confidence**: Confidence percentage
    - **probabilities**: Probabilities for both classes
    """
    try:
        # Validate file type
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in valid_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {valid_ext}"
            )
        
        # Load model and transforms
        model, transform = load_model()
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Prepare input
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)
        
        prediction = "Real" if pred.item() == 1 else "Fake"
        confidence = float(conf.item() * 100)
        
        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "probabilities": {
                "fake": float(probabilities[0, 0].item() * 100),
                "real": float(probabilities[0, 1].item() * 100)
            },
            "device": str(DEVICE)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/test", tags=["Testing"])
async def test_accuracy():
    """
    Test model accuracy on the test dataset
    
    Returns:
    - **accuracy**: Overall accuracy percentage
    - **precision**: Precision score
    - **recall**: Recall score
    - **f1_score**: F1 score
    - **confusion_matrix**: TP, TN, FP, FN
    """
    try:
        model, transform = load_model()
        
        test_dir = os.path.join(PROJECT_DIR, 'archive (1)', 'dataset', 'test')
        
        if not os.path.exists(test_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Test dataset not found at {test_dir}"
            )
        
        fake_dir = os.path.join(test_dir, 'Fake')
        real_dir = os.path.join(test_dir, 'Real')
        
        predictions = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        total = 0
        correct = 0
        
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
        
        # Test Fake images
        if os.path.exists(fake_dir):
            for img_file in os.listdir(fake_dir):
                if os.path.splitext(img_file)[1] in valid_ext:
                    try:
                        img_path = os.path.join(fake_dir, img_file)
                        img = Image.open(img_path).convert('RGB')
                        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
                        
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
                    except:
                        continue
        
        # Test Real images
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if os.path.splitext(img_file)[1] in valid_ext:
                    try:
                        img_path = os.path.join(real_dir, img_file)
                        img = Image.open(img_path).convert('RGB')
                        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
                        
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
                    except:
                        continue
        
        # Calculate metrics
        accuracy = (correct / total * 100) if total > 0 else 0
        precision = (predictions['TP'] / (predictions['TP'] + predictions['FP']) * 100) if (predictions['TP'] + predictions['FP']) > 0 else 0
        recall = (predictions['TP'] / (predictions['TP'] + predictions['FN']) * 100) if (predictions['TP'] + predictions['FN']) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        return {
            "total_images": total,
            "correct_predictions": correct,
            "accuracy": f"{accuracy:.2f}%",
            "precision": f"{precision:.2f}%",
            "recall": f"{recall:.2f}%",
            "f1_score": f"{f1:.2f}",
            "confusion_matrix": {
                "true_positives": predictions['TP'],
                "true_negatives": predictions['TN'],
                "false_positives": predictions['FP'],
                "false_negatives": predictions['FN']
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during testing: {str(e)}")

# --- RUN APPLICATION ---
if __name__ == '__main__':
    import uvicorn
    print("🚀 Starting MedSecure API...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔄 Alternative docs: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)
