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

def save_model(model, path):
    torch.save(model.state_dict(), path)

def main(args):
    print("Registering Data...")
    dataset = CatDataset(directory=args.directory)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Finish the Loading!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.head = torch.nn.Identity()
    model = model.to(device)
    triplet_loss = TripletMarginLoss(margin=1.0, p=2)
    optimizer = Adam(model.parameters(), lr=0.0001)

    model.train()
    print("Train begins...")
    for epoch in range(args.epochs):
        for data in dataloader:
            optimizer.zero_grad()

            # Unpack the data
            anchors, positives, negatives = data

            # Apply model to each set of images, add batch dimension before model application
            anchor_features = torch.cat([model(anchor.unsqueeze(0).to(device)) for sublist in anchors for anchor in sublist])
            positive_features = torch.cat([model(positive.unsqueeze(0).to(device)) for sublist in positives for positive in sublist])
            negative_features = torch.cat([model(negative.unsqueeze(0).to(device)) for sublist in negatives for negative in sublist])

            # Compute the triplet loss
            loss = triplet_loss(anchor_features, positive_features, negative_features)
            loss.backward()
            optimizer.step()

            # Print loss for the current batch
            print(f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{args.epochs} completed.")
    
    save_model(model, '/home/hyunseo/molo/model_weights.pth')
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a model on cat identification")
    parser.add_argument('--directory', type=str, default='/data/etc/molo/CAT_00', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()
    main(args)