import torch
import torchvision
import argparse
import time
from tqdm import tqdm
from datetime import datetime
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

    print(f"Train begins at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    for epoch in tqdm(range(args.epochs), desc="Epoch Progress"):
        total_loss = 0
        for images in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs} ", leave=False):
            optimizer.zero_grad()

            # Each have same num of augmentations
            # For example, anchor have 5 augmentations of (4,3,256,256) images
            # as list (batch_size, channel, img_size, img_size)
            anchors, positives, negatives = images

            # Apply model to each set of anchors, positives, and negatives for each cat
            anchor_features = torch.cat([model(anchor.unsqueeze(0).to(device)) for a in anchors for anchor in a])
            positive_features = torch.cat([model(positive.unsqueeze(0).to(device)) for p in positives for positive in p])
            negative_features = torch.cat([model(negative.unsqueeze(0).to(device)) for n in negatives for negative in n])


            # Compute the triplet loss
            loss = triplet_loss(anchor_features, positive_features, negative_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            #print(f"Loss: {loss.item():.4f}, Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"Epoch {epoch+1}/{args.epochs} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1} / {args.epochs} completed, Average Loss : {avg_loss:.4f}')

    torch.save(model, f'{args.output_dir}/output_model.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a model on cat identicifation")
    parser.add_argument('--directory', type=str, default='/data/etc/molo/CAT_train', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--debug', type= bool, default=False )
    parser.add_argument('--output_dir', type=str)


    args = parser.parse_args()
    main(args)

