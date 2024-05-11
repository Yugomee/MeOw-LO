import os
import torch
import torchvision
from sklearn.metrics.pairwise import cosine_similarity

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

