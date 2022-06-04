from torch import nn

class SignalModel(nn.Module):
    def __init__(self):
        super(SignalModel, self).__init__()
        self.layer_1 = nn.Linear(1, 1024, bias=False)
        self.layer_2 = nn.Linear(1024, 64, bias=False)
        self.layer_3 = nn.Linear(64, 1, bias=False)
        self.activation = nn.Tanh()

    def forward(self, X):
        X = self.layer_1(X)
        X = self.layer_2(X)
        X = self.layer_3(X)
        X = self.activation(X)
        return X