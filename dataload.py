import torch
import os
import pandas as pd
import cv2
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.utils import save_image

#TODO(Jeehyun) : Randomrotation cause all values returns to 0.
def get_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        #v2.RandomRotation(degrees=(0,10)),
        v2.Resize(256),
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
        landmark = (frame.to_numpy()[0][1:-1]).reshape((-1, 2))
        top_left = landmark[4]
        top_right = landmark[7]
        width = int(top_right[0] - top_left[0])
        height = width
        margin = int(width * 0.25)
        x = int(top_left[0]) - margin
        y = int(top_left[1]) - margin

        # Crop the image to the bounding box
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(x + width + 2 * margin, image.shape[1])
        y_end = min(y + height + 2 * margin, image.shape[0])

        crop_img = image[y_start:y_end, x_start:x_end]

        from IPython import embed; embed(colors="neutral")  # XXX DEBUG  # yapf: disable

        # Apply transformation
        if self.transform:
            crop_img = self.transform(crop_img)

        return crop_img








