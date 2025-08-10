import torch
import torch.nn as nn

from finworld.models.embed.base import Embed
from finworld.registry import EMBED

@EMBED.register_module(force=True)
class SparseEmbed(Embed):
    def __init__(self,
                 *args,
                 output_dim: int,
                 num_embeddings: int,
                 dropout: float = 0.0,
                 **kwargs):
        super(SparseEmbed, self).__init__(*args, **kwargs)

        self.output_dim = output_dim
        self.num_embeddings = num_embeddings

        self.embed = nn.Embedding(num_embeddings=num_embeddings,
                                  embedding_dim=output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.initialize_weights()

    def forward(self, x, **kwargs):
        """Forward pass of the embedding layer."""
        x = x.squeeze(-1)
        x = self.embed(x)
        x = self.dropout(x)

        return x

if __name__ == '__main__':
    device = torch.device("cpu")

    batch = torch.randint(0, 10, (2, 10, 1)).to(device)
    embed = SparseEmbed(num_embeddings=10, output_dim=64, dropout=0.1).to(device)
    output = embed(x=batch)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, output_dim)
    
    batch = torch.randint(0, 10, (2, 10)).to(device)
    embed = SparseEmbed(num_embeddings=10, output_dim=64, dropout=0.1).to(device)
    output = embed(x=batch)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, output_dim)
    
    batch = torch.randint(0, 10, (2, 1)).to(device)
    embed = SparseEmbed(num_embeddings=10, output_dim=64, dropout=0.1).to(device)
    output = embed(x=batch)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, output_dim)