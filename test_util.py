import os
import torch
import torchvision
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from PIL import Image

#TODO(Jeehyun) : should modify load_images to fit in ours

def load_images_pair(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
    images = [Image.open(file).convert('RGB') for file in files]
    return files, images

def extract_features(model, images, device):
    model.eval()
    images = torch.stack([image for image in images]).to(device)
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

def similarity_calculator(model, image1, image2, device):
    '''
    Args:
    image1, image2
    => feature1, feature2 (torch.Tensor): Feature vectors of shape (1, 2048)

    Returns:
    float: Cosine similarity between feature1 and feature2
    '''
    features = extract_features(model, [image1, image2], device)

    # feature1, feature2 지정
    feature1 = features[0]
    feature2 = features[1]

    # Calculate Cosine Similarity
    similarity = cosine_similarity([feature1], [feature2])[0][0]
    return similarity


def test_image_crop(image_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(image)

    # Filter detections to only 'cat' class (assuming 'cat' is class index 15)
    cat_detections = results.xyxy[0][results.xyxy[0][:, 5] == 15]  # Filter class ID

    if len(cat_detections) == 0: #If there is nothing detected, just pass the original
        print(f"No cats detected in {image_path.name}.")
        cropped_image = image

    # Select the detection with the highest confidence score
    top_cat = cat_detections[cat_detections[:, 4].argmax()]

        # Crop the image around the detected cat
    x1, y1, x2, y2 = int(top_cat[0]), int(top_cat[1]), int(top_cat[2]), int(top_cat[3])
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image
