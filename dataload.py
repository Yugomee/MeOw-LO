import torch
import os
import pandas as pd
import numpy as np
import cv2
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

def get_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=(0,45)),
        v2.RandomPerspective(distortion_scale=0.6, p=0.5),
        v2.Resize((256,256)),
        v2.ToDtype(torch.float32, scale=True)
    ])



class CatDataset(Dataset):
    def __init__(self, directory, transform=None, num_augmentations = 10):
        """
        Args:
            directory (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform if transform else get_transform()
        self.num_augmentations = num_augmentations

        self.filenames =[]
        # Dealing with no annotation
        for f in os.listdir(directory):
            if f.endswith('.jpg'):
                annotation_file = os.path.join(directory, f + '.cat')
                if os.path.isfile(annotation_file):  # Check if the annotation file exists
                    self.filenames.append(f)  # Only add the image if the annotation exists


    def __len__(self):
        return len(self.filenames) * self.num_augmentations

    def __getitem__(self, idx):
        file_idx = idx // self.num_augmentations
        img_name = os.path.join(self.directory, self.filenames[file_idx])
        annotation_name = img_name + '.cat'

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

         # Load annotations and compute the bounding box
        frame = pd.read_csv(annotation_name, sep =' ', header=None)
        landmarks = (frame.to_numpy()[0][1:-1]).reshape((-1, 2))


        # Calculate the angle of rotation
        left_ear = landmarks[4]
        right_ear = landmarks[7]
        angle = np.degrees(np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0]))

        # Calculate the center for rotation
        rotation_center_x,rotation_center_y = landmarks[6]
        rotation_matrix = cv2.getRotationMatrix2D((rotation_center_x, rotation_center_y), angle, 1)

        # Perform rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        # Update landmarks after rotation
        landmarks_homogenous = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])  # For affine transformation
        rotated_landmarks = rotation_matrix.dot(landmarks_homogenous.T).T

        # Recalculate the bounding box on the rotated image
        new_left_ear = rotated_landmarks[4]
        new_right_ear = rotated_landmarks[7]
        width = int(new_right_ear[0] - new_left_ear[0])
        margin = int(width * 0.25)
        height = width
        x = int(new_left_ear[0]) - margin
        y = int(new_left_ear[1]) - margin

        # Ensure cropping is within the image boundaries
        x = max(0, x)
        y = max(0, y)
        x_end = min(rotated_image.shape[1], x + width + 2 * margin)
        y_end = min(rotated_image.shape[0], y + height + 2 * margin)
        crop_img = rotated_image[y:y_end, x:x_end]

        if crop_img is None or crop_img.shape[0]==0 or crop_img.shape[1]==0:
            print(f'Invalid image encountered : {img_name}, {crop_img.shape}')


        # Apply transformation
        transformed_images = [self.transform(crop_img) for _ in range(self.num_augmentations)]


        #To debug...
        #trans_crop_img = to_pil_image(crop_img)
        #trans_crop_img.save(f'/home/jeehyun/coursework/DL/MeOw-LO/debug/test_{file_idx}_{idx}_{i}.png')

        return transformed_images








