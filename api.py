from fastapi import FastAPI
import torch
import random
from dataset import test_dataset
from model import MNISTModel
import uvicorn
import argparse

app = FastAPI()

model_path = "models/MNIST_model.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model = MNISTModel(1, 10)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@app.get("/predict")
async def predict():
    idx = random.randint(0, len(test_dataset) - 1)
    image, true_label = test_dataset[idx]
    
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted_class = torch.argmax(output, dim=1).item()
    
    return {
        "predicted_class": predicted_class,
        "true_label": true_label,
        "sample_index": idx
    }

@app.get("/")
async def root():
    return {"message": "MNIST Model API", "endpoints": ["/predict"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI MNIST Model Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )