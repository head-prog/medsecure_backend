# MedSecure - Medicine Detection API

A machine learning API for detecting counterfeit medicines using ResNet50 deep learning model.

## Features

- **Real-time Predictions**: Upload medicine images to detect if they're Real or Fake
- **High Accuracy**: 98.44% accuracy on test dataset
- **FastAPI**: Modern, fast web framework with automatic API documentation
- **Interactive Docs**: Built-in Swagger UI and ReDoc documentation
- **Accuracy Testing**: Endpoint to test model performance on full test dataset

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 98.44% |
| Precision | 99.65% |
| Recall | 97.91% |
| F1-Score | 98.77 |

## API Endpoints

### Status
- `GET /` - Welcome & API information
- `GET /health` - Health check
- `GET /model-info` - Model information and performance metrics

### Prediction
- `POST /predict` - Upload image for prediction

### Testing
- `GET /test` - Run accuracy test on test dataset

### Documentation
- `GET /docs` - Swagger UI (interactive documentation)
- `GET /redoc` - ReDoc alternative documentation

## Deployment on Render

### Prerequisites
- GitHub account
- Render account (https://render.com)
- This repository pushed to GitHub

### Steps to Deploy

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Connect to Render**
   - Go to https://render.com
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

3. **Configure Service**
   - **Name**: `medsecure-api`
   - **Root Directory**: Leave blank
   - **Runtime**: Python 3.10
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy
   - Your API will be live at `https://your-service-name.onrender.com`

### Alternative: Using render.yaml

If `render.yaml` is present, Render will use it automatically:
```bash
git push origin main
```

Your service will be deployed with the configuration from `render.yaml`.

## Local Development

### Setup
```bash
# Create virtual environment
python -m venv venv_torch

# Activate (Windows)
venv_torch\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run API
```bash
python api.py
```

API will be available at `http://localhost:8000`

### Run with image prediction
```bash
python MAIN.py
# Or provide image path
python MAIN.py "path/to/image.jpg"
```

### Test Model Accuracy
```bash
python MAIN.py test
```

## Usage Examples

### Using cURL
```bash
# Health check
curl https://your-service.onrender.com/health

# Get model info
curl https://your-service.onrender.com/model-info

# Upload image for prediction
curl -X POST "https://your-service.onrender.com/predict" \
  -F "file=@medicine_image.jpg"

# Test accuracy
curl https://your-service.onrender.com/test
```

### Using Python
```python
import requests

# Predict
with open('medicine.jpg', 'rb') as f:
    response = requests.post(
        'https://your-service.onrender.com/predict',
        files={'file': f}
    )
    print(response.json())

# Test accuracy
response = requests.get('https://your-service.onrender.com/test')
print(response.json())
```

## Files Structure

```
MEDSECURE/
├── api.py              # FastAPI application
├── MAIN.py             # Standalone prediction script
├── best_model.pth      # Trained PyTorch model
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── build.sh            # Build script
├── Procfile            # Heroku/alternative config
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Notes

- The model expects 224x224 RGB images
- Supported formats: JPG, JPEG, PNG, BMP
- Model runs on CPU or GPU automatically
- Predictions include confidence percentage

## Support

For issues or questions, please check the API documentation at `/docs` endpoint.
# medsecure_backend
