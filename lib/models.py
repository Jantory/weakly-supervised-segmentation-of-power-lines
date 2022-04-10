

#
#       NN model file
#   Author: Louis Lettry
#

import torch as T

#   Too simple architecture - but indicating how it works
class TestSDFArchitecture(T.nn.Module):
    def __init__(self):
        super().__init__()

        inner_size = 16

        #   Create layers & weights
        self.lin1 = T.nn.Linear(3, inner_size) #    Dense/Fully connected layer
        self.relu1 = T.nn.ReLU()    #   Followed by its non linearity

        self.lin2 = T.nn.Linear(inner_size, inner_size)
        self.relu2 = T.nn.ReLU()

        self.lin3 = T.nn.Linear(inner_size, 1)

    #   Make a forward pass
    def forward(self, x):
        relin1 = self.relu1(self.lin1(x))
        relin2 = self.relu2(self.lin2(relin1))
        result = self.lin3(relin2)
        return result

#   Too simple architecture - but indicating how it works
class SixLinear(T.nn.Module):
    def __init__(self):
        super().__init__()

        inner_size = 256

        #   Create layers & weights
        self.lin1 = T.nn.Linear(3, inner_size) #    Dense/Fully connected layer
        self.relu1 = T.nn.ReLU()    #   Followed by its non linearity

        self.lin2 = T.nn.Linear(inner_size, inner_size)
        self.relu2 = T.nn.ReLU()

        self.lin3 = T.nn.Linear(inner_size, inner_size)
        self.relu3 = T.nn.ReLU()

        self.lin4 = T.nn.Linear(inner_size, inner_size)
        self.relu4 = T.nn.ReLU()

        self.lin5 = T.nn.Linear(inner_size, inner_size)
        self.relu5 = T.nn.ReLU()

        self.lin6 = T.nn.Linear(inner_size, 1)

    #   Make a forward pass
    def forward(self, x):
        relin1 = self.relu1(self.lin1(x))
        relin2 = self.relu2(self.lin2(relin1))
        relin3 = self.relu3(self.lin3(relin2))
        relin4 = self.relu4(self.lin4(relin3))
        relin5 = self.relu5(self.lin5(relin4))
        result = self.lin6(relin5)
        return result

#   Too simple architecture - but indicating how it works
class TenMidSkipLinear(T.nn.Module):
    def __init__(self):
        super().__init__()

        inner_size = 512

        #   Create layers & weights
        self.lin1 = T.nn.Linear(3, inner_size) #    Dense/Fully connected layer
        self.relu1 = T.nn.ReLU()    #   Followed by its non linearity

        self.lin2 = T.nn.Linear(inner_size, inner_size)
        self.relu2 = T.nn.ReLU()

        self.lin3 = T.nn.Linear(inner_size, inner_size)
        self.relu3 = T.nn.ReLU()

        self.lin4 = T.nn.Linear(inner_size, inner_size)
        self.relu4 = T.nn.ReLU()

        self.lin5 = T.nn.Linear(inner_size, inner_size)
        self.relu5 = T.nn.ReLU()

        #   Double this one to connect back input ? 
        self.lin6 = T.nn.Linear(inner_size + 3, inner_size)
        self.relu6 = T.nn.ReLU()

        self.lin7 = T.nn.Linear(inner_size, inner_size)
        self.relu7 = T.nn.ReLU()

        self.lin8 = T.nn.Linear(inner_size, inner_size)
        self.relu8 = T.nn.ReLU()

        self.lin9 = T.nn.Linear(inner_size, inner_size)
        self.relu9 = T.nn.ReLU()

        self.lin10 = T.nn.Linear(inner_size, 1)

    #   Make a forward pass
    def forward(self, x):
        relin1 = self.relu1(self.lin1(x))
        relin2 = self.relu2(self.lin2(relin1))
        relin3 = self.relu3(self.lin3(relin2))
        relin4 = self.relu4(self.lin4(relin3))
        relin5 = self.relu5(self.lin5(relin4))
        
        concat = T.cat((relin5, x), dim = 1)

        relin6 = self.relu6(self.lin6(concat))
        relin7 = self.relu7(self.lin7(relin6))
        relin8 = self.relu8(self.lin8(relin7))
        relin9 = self.relu9(self.lin9(relin8))
        result = self.lin10(relin9)
        
        return result

NN_models = {"test":TestSDFArchitecture, "6lin":SixLinear, "10MSlin":TenMidSkipLinear}