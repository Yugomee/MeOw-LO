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

class Encoder(nn.Module):
    def __init__(self, pretrained=True, model_name="google/vit-base-patch16-224-in21k", embed_dim=128):
        super(Encoder, self).__init__()
        self.pretrained = pretrained

        # Load the pre-trained model or initialize a new one
        if self.pretrained:
            self.backbone = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig(image_size=224, num_channels=3, hidden_size=768)  # Default values for ViT
            self.backbone = ViTModel(config)

        # Redefine the classifier head to output 128-dimensional embeddings
        self.cls_head = nn.Linear(self.backbone.config.hidden_size, embed_dim)

    def forward(self, x):
        # Forward pass through the backbone to get the transformer's output
        outputs = self.backbone(x)
        # Extract the [CLS] token's output (assuming it's the first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Pass the [CLS] token's output through the classifier head to get the final embedding
        embedding = self.cls_head(cls_output)
        return embedding


def save_model(model, path):
    torch.save(model.state_dict(), path)

def main(args):
    print("Registering Data...")
    dataset = CatDataset(directory=args.directory)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Finish the Loading!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(pretrained=True, embed_dim=128)
    model = model.to(device)
    triplet_loss = TripletMarginLoss(margin=1.0, p=2)
    optimizer = Adam(model.parameters(), lr=0.0001)

    model.train()
    print("Train begins...")
    for epoch in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            anchors, positives, negatives = batch

            # 각 이미지 리스트를 텐서로 변환
            anchor_tensors = [torch.stack([img.to(device) for img in sublist]) for sublist in anchors]
            positive_tensors = [torch.stack([img.to(device) for img in sublist]) for sublist in positives]
            negative_tensors = [torch.stack([img.to(device) for img in sublist]) for sublist in negatives]
            # print(anchor_tensors[0].shape, positive_tensors[0].shape, negative_tensors[0].shape)
            # torch.Size([16, 3, 224, 224]) torch.Size([16, 3, 224, 224]) torch.Size([16, 3, 224, 224])

            # 모델을 통해 특성 계산
            anchor_features = torch.cat([model(anchor) for anchor in anchor_tensors])
            positive_features = torch.cat([model(positive) for positive in positive_tensors])
            negative_features = torch.cat([model(negative) for negative in negative_tensors])
            # print(anchor_features.shape, positive_features.shape, negative_features.shape)
            # torch.Size([80, 128]) torch.Size([80, 128]) torch.Size([80, 128])

            # 손실 계산
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
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    args = parser.parse_args()
    main(args)