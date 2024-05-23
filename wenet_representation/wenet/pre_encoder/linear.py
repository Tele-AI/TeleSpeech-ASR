import torch

class LinearProjection(torch.nn.Module):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.drop_out = torch.nn.Dropout(dropout)
        self.linear_layer = torch.nn.Linear(input_size, output_size)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain("linear")
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ):
        output = self.linear_layer(self.drop_out(input))
        return output, input_lengths
