import torch
from torch.utils.data import DataLoader
from dataset import VOGMaze2dOfflineRLDataset
from bvae import BetaVAE
from tqdm import tqdm

# configs
config = {
    "num_epochs": 20,
    "batch_size": 4096,
    "lr": 1e-5,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "name": "beta_vae",
    "load": None
}


# set seed
torch.manual_seed(config["seed"])
if config["device"] == "cuda":
    torch.cuda.manual_seed(config["seed"])

#initialize wandb
import wandb
wandb.init(project='hs_vae', entity='Hierarchical-Diffusion-Forcing', config=config)

# Load the dataset
train_dataset = VOGMaze2dOfflineRLDataset(split='training')
validation_dataset = VOGMaze2dOfflineRLDataset(split='validation')

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize the model
model = BetaVAE()
model.to(config["device"])

# Load the model if specified
if config["load"]:
    model.load_state_dict(torch.load(config["load"]))

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# Train
for epoch in tqdm(range(config["num_epochs"])):
    model.train()
    for i, batch in enumerate(train_loader):
        batch = batch["observations"].to(config["device"])
        optimizer.zero_grad()
        recons, inputs, mu, log_var = model(batch)
        loss_dict = model.loss_function(recons, inputs, mu, log_var)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Iter: {model.num_iter}", end=" " )
            for k, v in loss_dict.items():
                wandb.log({k: v})
                print(f"{k}: {v}", end=", ")
            print(end="\r")
        
    # Validation
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            pos = batch["pos"]
            batch = batch["observations"]
            batch = batch.to(config["device"])
            recons, inputs, mu, log_var = model(batch)
            loss_dict = model.loss_function(recons, inputs, mu, log_var)
            if i == 0:
                # select random indicies 
                idx = torch.randint(0, batch.size(0), (8,))

                base = idx[0]
                emb_dist_list = []
                pos_dist_list = []
                for i in idx[1:]:
                    emb_dist = torch.norm(mu[base] - mu[i])
                    pos_dist = torch.norm(pos[base] - pos[i])
                    emb_dist_list.append(emb_dist)
                    pos_dist_list.append(pos_dist)
                # calculate the correlation between the distances
                emb_dist_list = torch.tensor(emb_dist_list)
                pos_dist_list = torch.tensor(pos_dist_list)
                corr = torch.corrcoef(emb_dist_list, pos_dist_list)
                wandb.log({"corr": corr[0, 1]})
                print("Correlation between the distances:", corr[0, 1])

                # log the base input and reconstruction
                inputs = inputs[idx]
                recons = recons[idx]
                inputs = inputs * 71.0288272312382 + 141.785487953533
                recons = recons * 71.0288272312382 + 141.785487953533
                wandb.log({"input": wandb.Image(inputs)})
                wandb.log({"recons": wandb.Image(recons)})

                
    for k, v in loss_dict.items():
        wandb.log({k+'val': v})
        print("Epoch:", epoch ,k+'val', ":", v, end=", ")

    # Save the model with time
    import datetime
    torch.save(model.state_dict(), f"beta_vae_{datetime.datetime.now()}.pth")

