### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Iterable

### External Imports ###
import torch as tc
import torch.nn.functional as F
import torchsummary as ts
import torchvision as tv

### Internal Imports ###

import building_blocks as bb

########################






def config_lung_efficientnet() -> dict:
    ### Define Params ###
    backend_name = "efficientnet"
    input_features = 1280

    num_classes = 1
    num_outputs = 1
    num_external_features = 16
    ### Parse ###
    config = {}
    config['backend_name'] = backend_name
    config['input_features'] = input_features
    config['num_classes'] = num_classes
    config['num_external_features'] = num_external_features
    config['num_outputs'] = num_outputs
    return config


class MILNet(tc.nn.Module):
    def __init__(self, backend_name, input_features, num_classes, num_external_features, num_outputs):
        super(MILNet, self).__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.backend_name = backend_name
        self.num_external_features = num_external_features

        if self.backend_name == "efficientnet":
            self.E = 1280
            self.L = self.E
            self.D = 256
            self.K = self.num_classes
        elif self.backend_name == "efficientnet_m":
            self.E = 1280
            self.L = self.E
            self.D = 256
            self.K = self.num_classes
        elif self.backend_name == "resnet18":
            self.E = 512
            self.L = self.E
            self.D = 256
            self.K = self.num_classes
        elif self.backend_name == "resnet50":
            self.E = 2048
            self.L = self.E
            self.D = 256
            self.K = self.num_classes
        else:
            raise ValueError("Unsupported backend.")
            
        self.attention = tc.nn.Sequential(
            tc.nn.Linear(self.L, self.D),
            tc.nn.Tanh(),
            tc.nn.Linear(self.D, self.K)
        )

        self.classifier = tc.nn.Sequential(
            tc.nn.Linear(self.L*self.K + self.num_external_features, 128),
            tc.nn.PReLU(),
            tc.nn.Linear(128, num_outputs)
        )


    def forward(self, x, features):
        x = x.view(-1, self.input_features)
        A = self.attention(x)
        A = tc.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = tc.mm(A, x)
        output = self.classifier(tc.cat((M, features), dim=1))
        return output



class MILNetExtended(tc.nn.Module):
    def __init__(self, backend_name, input_features, num_classes, num_external_features, num_outputs):
        super(MILNetExtended, self).__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.backend_name = backend_name
        self.num_external_features = num_external_features

        if self.backend_name == "efficientnet":
            self.E = 1280
            self.L = self.E
            self.D = 256
            self.K = self.num_classes
        elif self.backend_name == "efficientnet_m":
            self.E = 1280
            self.L = self.E
            self.D = 256
            self.K = self.num_classes
        elif self.backend_name == "resnet18":
            self.E = 512
            self.L = self.E
            self.D = 256
            self.K = self.num_classes
        elif self.backend_name == "resnet50":
            self.E = 2048
            self.L = self.E
            self.D = 256
            self.K = self.num_classes
        else:
            raise ValueError("Unsupported backend.")
        
        self.intermediate_layer = tc.nn.Sequential(
            tc.nn.Linear(self.L + self.num_external_features, self.L),
            tc.nn.GroupNorm(64, self.L),
            tc.nn.PReLU(),
            tc.nn.Linear(self.L, self.L),
            tc.nn.GroupNorm(64, self.L),
            tc.nn.PReLU(),
        )
  
        self.attention = tc.nn.Sequential(
            tc.nn.Linear(self.L, self.D),
            tc.nn.Tanh(),
            tc.nn.Linear(self.D, self.K)
        )

        self.classifier = tc.nn.Sequential(
            tc.nn.Linear(self.L*self.K, num_outputs)
        )


    def forward(self, x, features):
        x = x.view(-1, self.input_features)
        embeddings = self.intermediate_layer(tc.cat((x, features.repeat(x.shape[0], 1)), dim=1))
        A = self.attention(embeddings)
        A = tc.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = tc.mm(A, embeddings)
        output = self.classifier(M)
        return output


def run():
    pass

if __name__ == "__main__":
    run()