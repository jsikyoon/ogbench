import torch
from torch.utils.data import DataLoader
from dataset import VOGMaze2dOfflineRLDataset
from bvae import BetaVAE
from tqdm import tqdm
import argparse

# configs
parser = argparse.ArgumentParser(description='Beta VAE Training Configuration')

parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--name', type=str, default='beta_vae', help='Model name')
parser.add_argument('--load', type=str, default="loss0.09170543402433395_1e-10_lr_0.0001_190.pth", help='Path to load model')
parser.add_argument('--kld_weight', type=float, default=1e-6, help='KLD weight')
parser.add_argument('--group_name', type=str, default='eval', help='KLD weight')


args = parser.parse_args()
config = vars(args)


# set seed
torch.manual_seed(config["seed"])
if config["device"] == "cuda":
    torch.cuda.manual_seed(config["seed"])

#initialize wandb
import wandb
wandb.init(project='hs_vae', entity='Hierarchical-Diffusion-Forcing', config=config)

# Load the dataset
validation_dataset = VOGMaze2dOfflineRLDataset(split='validation')

# Create the dataloaders
validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False)

print("Dataset loaded")

# Initialize the model
model = BetaVAE()
model.to(config["device"])

# Load the model if specified
if config["load"]:
    model.load_state_dict(torch.load(config["load"]))

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])


# Validation
model.eval()
with torch.no_grad():
    corr_list = []
    recon_loss_list = []
    for i, (obs, pos) in enumerate(validation_loader):            
        obs = obs.to(config["device"])
        recons, inputs, mu, log_var = model(obs)
        loss_dict = model.loss_function(recons, inputs, mu, log_var)
        recon_loss_list.append(loss_dict["Reconstruction_Loss"])
        pos = pos.to(config["device"])

        # select random indicies 
        emb_dist_list = []
        pos_dist_list = []
        for i in range(1, obs.size(0)):
            emb_dist = torch.norm(mu[0] - mu[i])
            pos_dist = torch.norm(pos[0] - pos[i])
            emb_dist_list.append(emb_dist)
            pos_dist_list.append(pos_dist)
        
        # calculate the correlation between the distances
        emb_dist_list = torch.tensor(emb_dist_list)
        pos_dist_list = torch.tensor(pos_dist_list)
        
        # corr = torch.corrcoef(emb_dist_list, pos_dist_list)
        # wandb.log({"corr": corr[0, 1]})
        try:
            corr = torch.corrcoef(torch.stack([emb_dist_list, pos_dist_list]))
            corr_list.append(corr[0, 1])
        except:
            pass
        if i == 0:
            idx = torch.randint(0, obs.size(0), (8,))
            base = idx[0]
            # log the base input and reconstruction
            inputs = inputs[idx[:8]]
            recons = recons[idx[:8]]
            inputs = inputs * 71.0288272312382 + 141.785487953533
            recons = recons * 71.0288272312382 + 141.785487953533
            wandb.log({"input": wandb.Image(inputs)})
            wandb.log({"recons": wandb.Image(recons)})

    corr_list = torch.tensor(corr_list)
    corr_mean = torch.mean(corr_list)
    corr_std = torch.std(corr_list)
    wandb.log({"corr_mean": corr_mean, "corr_std": corr_std})
    print("Correlation mean:", corr_mean)


    recon_loss_list = torch.tensor(recon_loss_list)
    recon_loss_mean = torch.mean(recon_loss_list)
    wandb.log({"recon_loss_mean": recon_loss_mean})