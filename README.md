# Fire, Smoke, and Non fire Classifier
## Problem Statement
Wildfires and accidental fires cause significant damage to lives, property, and the environment. Early detection of fire and smoke through automated image classification can dramatically improve response times and reduce damage. However, differentiating between fire, smoke, and non-hazardous scenes in images is challenging due to differences in lighting conditions and backgrounds.
<br><br>
This model addresses this challenge by using deep learning with PyTorch to classify images into three categories: fire, smoke, and non fire. This solution can help emergency responders and fire monitoring systems quickly and accurately identify potential fire hazards, which allows for faster intervention.


## Project Structure
```md
├── notebook.ipynb # EDA and model experimentation
├── train.py # Model training script
├── export.py # Export trained model to ONNX
├── predict.py # FastAPI inference service
├── test-api.py # API testing script
├── test-image.jpg # Sample testing image
├── smoke_fire_classifier.onnx
├── smoke_fire_classifier.pth
├── Dockerfile # Container configuration
├── pyproject.toml
└── README.md
```

## Dataset
Link: https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
<br>
Since the original dataset is very large, a subset of the dataset was created by randomly selecting 4,000 images per class from the training set (12,000 total images). Images were not included in the repository, and should be saved locally under images/train_small and images/test. Since the images are randomly selected, results may be slightly different, but the procedure remains the same.
<br><br>
The original test set provided by the dataset was left unchanged and used for final evaluation.


## Model Training and Performance
Details can be found in notebook.ipynb. The EfficientNet-B0 model was used, with key parameters including learning rate, inner size, and dropout rate tuned. The model was evaluated on the test dataset with an accuracy of **96.75%**.

## Local Setup
1. Clone repository

2. Create a virtual environment:
```
uv venv .venv
```

3. Activate the virtual environment:
```
source .venv/Scripts/activate
```

4. Install dependencies:
```
uv sync
```

5. Train model
```
python train.py
python export.py
```

## Docker Setup
1. Build Docker Image
```
docker build -t fire-smoke-classifier .
```

2. Run Docker Container
```
docker run -it --rm -p 9696:9696 fire-smoke-classifier
```

The API will then be accessible at http://localhost:9696/predict.

Test prediction using sample data:
```
python test-api.py
```

Or

```
curl -X POST \
  http://localhost:9696/predict \
  -H "accept: application/json" \
  -F "file=@test-image.jpg"
```

Response
```
{
    'predicted_class': 'fire', 
    'confidence': 0.9994840621948242
}
```

Push docker image to DockerHub:
```
docker tag fire-smoke-classifier:latest <username>/fire-smoke-classifier:latest

docker push <username>/fire-smoke-classifier:latest
```

## Deploy to cloud
The docker image was deployed to the cloud using [Claw Cloud Run](https://us-east-1.run.claw.cloud/). 
1. Go to App Launchpad -> Create.
2. Under image, include docker image name: ```<dockerhub username>/fire-smoke-classifier:latest```
3. Under network, update container port to ```9696``` and allow public access

Live Demo: https://smoke-fire-classifier.netlify.app/