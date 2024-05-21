import torch
import torch.nn as nn
import torchvision
import argparse
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from dataload import CatDataset
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.optim import Adam
from torch.nn import TripletMarginLoss
from transformers import ViTModel, ViTConfig
import wandb
import time

class Encoder(nn.Module):
    def __init__(self, pretrained=True, model_name="google/vit-base-patch16-224-in21k", embed_dim=128):
        super(Encoder, self).__init__()
        if pretrained:
            self.backbone = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig(image_size=224, num_channels=3, hidden_size=768)
            self.backbone = ViTModel(config)
        self.cls_head = nn.Linear(self.backbone.config.hidden_size, embed_dim)

    def forward(self, x):
        outputs = self.backbone(x)
        cls_output = outputs.last_hidden_state[:, 0, :]
        embedding = self.cls_head(cls_output)
        return embedding

def save_model(model, path):
    torch.save(model.state_dict(), path)

def main(args):
    wandb.init(project="cat-identification", entity="khsvv", config=args)
    print("Registering Data...")
    dataset = CatDataset(directory=args.directory)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Finish the Loading!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(pretrained=True, embed_dim=128).to(device)
    triplet_loss = TripletMarginLoss(margin=1.0, p=2)
    optimizer = Adam(model.parameters(), lr=args.lr)

    model.train()
    print("Train begins...")
    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0.0  # Initialize total loss for the epoch
        for batch in dataloader:
            optimizer.zero_grad()
            anchors, positives, negatives = batch

            anchor_tensors = [torch.stack([img.to(device) for img in sublist]) for sublist in anchors]
            positive_tensors = [torch.stack([img.to(device) for img in sublist]) for sublist in positives]
            negative_tensors = [torch.stack([img.to(device) for img in sublist]) for sublist in negatives]
            #print(anchor_tensors[0].shape)

            anchor_features = torch.cat([model(anchor) for anchor in anchor_tensors])
            positive_features = torch.cat([model(positive) for positive in positive_tensors])
            negative_features = torch.cat([model(negative) for negative in negative_tensors])
            #print(anchor_features.shape)

            loss = triplet_loss(anchor_features, positive_features, negative_features)
            total_loss += loss.item()  # Accumulate loss
            loss.backward()
            optimizer.step()  # Perform optimization step after each batch

        epoch_duration = time.time() - start_time
        average_loss = total_loss / len(dataloader)  # Calculate the average loss for the epoch
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_duration:.2f} seconds with average loss: {average_loss:.4f}")
        wandb.log({"average_loss": average_loss, "epoch": epoch+1})

    save_model(model, '/home/hyunseo/molo/model_weights.pth')
    print("Model saved successfully.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a model on cat identification")
    parser.add_argument('--directory', type=str, default='/data/etc/molo/CAT_00', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')  # Adjusted type to float for learning rate
    args = parser.parse_args()
    main(args)
