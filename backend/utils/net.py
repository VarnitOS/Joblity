from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 50),
            nn.ReLU(),
            # nn.Linear(50, 50),
            # nn.ReLU(),
            nn.Linear(50, 30)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
        