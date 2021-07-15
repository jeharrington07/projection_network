import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.nn.modules.batchnorm import BatchNorm2d

#TODO: experiment with network parameters. convolutional network (see https://github.com/openai/random-network-distillation, maybe follow the architecture of toad-gan for this part)

#dropout of x initializes one weight in every x to 0
def create_random_dense(input_size, output_size, device, dropout):

    class RandomNet(nn.Module):
        def __init__(self, device):
            super(RandomNet, self).__init__()
            self.device = device
            self.linear = nn.Linear(in_features = input_size, out_features = output_size, bias = False)
            inds = [num for num in range(len(self.linear.weight.data[0])) if num % dropout == 0]
            inds = torch.tensor(inds)
            print(inds)
            self.linear.weight.data.index_fill_(1, inds, 0)
            print(self.linear.weight.data)

        def forward(self, input):
            #output = self.norm(self.main(input))
            #return output
            output = self.main(input)
            return output
    return RandomNet(device)


def random_fake(output_size):
    return torch.randn(output_size)