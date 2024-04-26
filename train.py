import torch
import torchvision
import argparse
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from dataload import CatDataset
import matplotlib.pyplot as plt


def main(args) :
    print("Registering Data...")
    dataset = CatDataset(directory=args.directory)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Finish the Loading!")


    #TrainLoop
    print("Train begins...")
    for epoch in range(args.epochs):
        for images in dataloader:
            #TODO(jeehyun) below is for debugging. it could be deprecated
            if args.debug :
                print(f'Batch size : {len(images)}')
                print(f'Tensor shape: {images.size()}')
                break
            pass
        print(f"Epoch {epoch+1}/{args.epochs} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a model on cat identicifation")
    parser.add_argument('--directory', type=str, default='/data/etc/molo/CAT_00', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--debug', type= bool, default=False )


    args = parser.parse_args()
    main(args)

