import torch
import torchvision
import argparse
import time
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from dataload import CatDataset
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.nn import TripletMarginLoss
from torch.nn.functional import cosine_similarity
from transformers import ViTModel, ViTConfig
import wandb

def extract_and_average_features(augmentations, model, device):
    # 'augmentations' is a list of 5 tensors, each tensor is [batch_size, channels, width, height]
    # Process each batch index across all augmentations
    # Initialize tensor to store averaged features for each cat in the batch
    batch_size = augmentations[0].shape[0]
    # Assume we know the feature dimension from the model, say feature_dim
    feature_dim = 128  # input size to model
    averaged_features = torch.zeros(batch_size, feature_dim, device=device)

    # Change the shape of input
    stacked_tensors = torch.stack(augmentations) #[num_aug, batch_size, 3, 224, 224]
    permuted_tensors = stacked_tensors.permute(1, 0, 2, 3, 4) # [batch_size, num_aug, 3, 224,224]
    permuted_tensors = permuted_tensors.to(device)

    averaged_features = []
    # Iterate over each cat in the batch
    for i in range(batch_size):
        # Collect features for the i-th cat across all augmentations
        aug_cat = permuted_tensors[i]
        cat_features = []
        for j in aug_cat: # j : [3,224,224]
            single_cat = j.unsqueeze(0)
            features = model(single_cat)  # Get features for this single cat image
            cat_features.append(features)
        stacked_features = torch.stack(cat_features)
        averaged_output = torch.mean(stacked_features, dim=0)
        averaged_features.append(averaged_output)

    averaged_features_tensor = torch.cat(averaged_features, dim=0)

    return averaged_features_tensor

def validate(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    positive_similarities = []
    negative_similarities = []

    with torch.no_grad():  # No gradients needed
        for batch in tqdm(dataloader, desc = 'Validation process'):
            anchors, positives, negatives = batch

            # Extract features without gradient calculation
            anchor_features = extract_and_average_features(anchors, model, device)
            positive_features = extract_and_average_features(positives, model, device)
            negative_features = extract_and_average_features(negatives, model, device)
            n_anchor_features = torch.nn.functional.normalize(anchor_features, p=2, dim=1)
            n_positive_features = torch.nn.functional.normalize(positive_features, p=2, dim=1)
            n_negative_features = torch.nn.functional.normalize(negative_features, p=2, dim=1)
            # Compute cosine similarity
            pos_similarity = cosine_similarity(n_anchor_features, n_positive_features)
            neg_similarity = cosine_similarity(n_anchor_features, n_negative_features)

             # Collect data for analysis
            positive_similarities.extend(pos_similarity.tolist())
            negative_similarities.extend(neg_similarity.tolist())

    # Compute averages of collected data
    pos_sim_avg = sum(positive_similarities) / len(positive_similarities) if positive_similarities else 0
    neg_sim_avg = sum(negative_similarities) / len(negative_similarities) if negative_similarities else 0

    return pos_sim_avg, neg_sim_avg


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
    #wandb.init(project="molo", entity="pai_jeehyun", config=args)
    print("Registering Data...")
    dataset = CatDataset(directory=args.directory)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = CatDataset(directory=args.test_directory)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=True)
    print("Finish the Loading!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(pretrained=True, embed_dim=128).to(device)
    triplet_loss = TripletMarginLoss(margin=3.0, p=2)
    optimizer = Adam(model.parameters(), lr=args.lr)

    model.train()
    print("Train begins...")
    for epoch in tqdm(range(args.epochs), desc = "Epochs"):
        total_loss = 0.0  # Initialize total loss for the epoch
        for batch in tqdm(dataloader, desc="Training Batch", leave=False):
            optimizer.zero_grad()
            anchors, positives, negatives = batch
            #print(f"Feature extracting starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            anchor_features = extract_and_average_features(anchors, model, device)
            positive_features = extract_and_average_features(positives, model, device)
            negative_features = extract_and_average_features(negatives, model, device)
            n_anchor_features = torch.nn.functional.normalize(anchor_features, p=2, dim=1)
            n_positive_features = torch.nn.functional.normalize(positive_features, p=2, dim=1)
            n_negative_features = torch.nn.functional.normalize(negative_features, p=2, dim=1)
            #print(f"Feature extracting ends..Loss calculates starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            loss = triplet_loss(n_anchor_features, n_positive_features, n_negative_features)

            #print(f"Loss calculates ends at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate loss

            #print(f"Loss: {loss.item():.4f}, Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"Epoch {epoch+1}/{args.epochs} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1} / {args.epochs} completed, Average Loss : {avg_loss:.4f}')

        #wandb.log({"average_loss": avg_loss, "epoch": epoch+1})

        pos_sim, neg_sim = validate(model, val_dataloader, device)
        print(f"Average Positive Similarity: {pos_sim:.4f}, Average Negative Similarity: {neg_sim:.4f}")
        #wandb.log({"positive similarity in validation": pos_sim, "epoch": epoch +1})
        #wandb.log({"nositive similarity in validation": neg_sim, "epoch": epoch +1})

    torch.save(model, f'{args.output_dir}/output_model_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt')

    print("Model saved successfully.")
    #wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a model on cat identification")
    parser.add_argument('--directory', type=str, default='/data/etc/molo/CAT_train', help='Directory containing the dataset')
    parser.add_argument('--test_directory', type=str, default='/data/etc/molo/CAT_val', help = 'Directory for validation set')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')  # Adjusted type to float for learning rate
    parser.add_argument('--output_dir', type=str, default = '/data/etc/molo')
    args = parser.parse_args()
    main(args)
