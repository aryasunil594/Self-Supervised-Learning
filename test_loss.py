import torch

from contrastive_loss import NTXentLoss


# Create random projections
z1 = torch.randn(4, 128)
z2 = torch.randn(4, 128)

# Create loss function
loss_function = NTXentLoss(temperature=0.5)

# Calculate loss
loss = loss_function(z1, z2)

print("Contrastive Loss:", loss.item())