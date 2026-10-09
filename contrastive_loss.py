import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):

        batch_size = z1.size(0)

        # Normalize representations
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Combine both views
        representations = torch.cat([z1, z2], dim=0)

        # Similarity matrix
        similarity_matrix = torch.matmul(
            representations,
            representations.T
        )

        similarity_matrix = similarity_matrix / self.temperature

        # Remove self-similarity
        mask = torch.eye(
            2 * batch_size,
            device=similarity_matrix.device
        ).bool()

        similarity_matrix.masked_fill_(mask, -9e15)

        # Positive pairs
        positive_indices = torch.arange(
            batch_size,
            device=z1.device
        )

        positive_indices = torch.cat([
            positive_indices + batch_size,
            positive_indices
        ])

        positives = similarity_matrix[
            torch.arange(2 * batch_size, device=z1.device),
            positive_indices
        ]

        # Denominator
        denominator = torch.logsumexp(
            similarity_matrix,
            dim=1
        )

        # NT-Xent loss
        loss = -positives + denominator

        return loss.mean()