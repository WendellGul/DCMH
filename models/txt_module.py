from torch import nn
from torch.nn import functional as F
from models.basic_module import BasicModule

LAYER1_NODE = 8192


class TxtModule(BasicModule):
    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = "text_model"

        # first full-connect layer (input_dim * 8192)
        self.fc1 = nn.Linear(y_dim, LAYER1_NODE)

        # second full-connect layer (8192 * bit)
        self.fc2 = nn.Linear(LAYER1_NODE, bit)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
