import torch
import torchvision
import argparse
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from dataload import CatDataset
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.optim import Adam
from torch.nn import TripletMarginLoss


def main(args) :
    print("Registering Data...")
    dataset = CatDataset(directory=args.directory)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Finish the Loading!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델 초기화
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.head = torch.nn.Identity()
    model = model.to(device)
    triplet_loss = TripletMarginLoss(margin=1.0, p=2)
    optimizer = Adam(model.parameters(), lr=0.0001)


    #TrainLoop
    model.train()

    print("Train begins...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        for images in dataloader:
            optimizer.zero_grad()

            # Each have same num of augmentations
            # For example, anchor have 5 augmentations of (4,3,256,256) images
            # as list (batch_size, channel, img_size, img_size)
            anchors, positives, negatives = images

            print("Check image size")
            print(f'Anchor : {anchors[0].shape}, Positive : {positives[0].shape}')

            # Move each tensor in the list to the device
            anchors = [anchor.to(device) for anchor in anchors]
            positives = [positive.to(device) for positive in positives]
            negatives = [negative.to(device) for negative in negatives]

            # Compute features for each set of images
            anchor_features = torch.stack([model(anchor) for anchor in anchors])
            positive_features = torch.stack([model(positive) for positive in positives])
            negative_features = torch.stack([model(negative) for negative in negatives])

            # Compute the triplet loss
            loss = triplet_loss(anchor_features, positive_features, negative_features)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a model on cat identicifation")
    parser.add_argument('--directory', type=str, default='/data/etc/molo/CAT_00', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--debug', type= bool, default=False )


    args = parser.parse_args()
    main(args)

