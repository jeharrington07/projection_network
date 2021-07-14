import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.nn.modules.batchnorm import BatchNorm2d

#TODO: experiment with network parameters. convolutional network (see https://github.com/openai/random-network-distillation, maybe follow the architecture of toad-gan for this part)
def create_random_dense(input_size, output_size, device):

    class RandomNet(nn.Module):
        def __init__(self, device):
            super(RandomNet, self).__init__()
            self.device = device
            self.main = nn.Sequential(
                nn.Linear(in_features = input_size, out_features = output_size, bias = False),
                nn.ReLU()
            )

        def forward(self, input):
            #output = self.norm(self.main(input))
            #return output

            output = self.main(input)
            return output
    return RandomNet(device)


def randomFake(output_size):
    return torch.randn(output_size)