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

from paths import paths as p
import building_blocks as bb

########################


def config_prostate_efficientnet(weights) -> dict:
    ### Define Params ###
    backend = tv.models.efficientnet_v2_s(weights=None)
    backend.load_state_dict(weights)
    backend_name = "efficientnet"
    input_features = 1280

    num_classes = 1
    num_outputs = 2
    num_external_features = 3
    ### Parse ###
    config = {}
    config['backend'] = backend
    config['backend_name'] = backend_name
    config['input_features'] = input_features
    config['num_classes'] = num_classes
    config['num_external_features'] = num_external_features
    config['num_outputs'] = num_outputs
    return config

def config_prostate_efficientnet2(weights) -> dict:
    ### Define Params ###
    backend = tv.models.efficientnet_v2_s(weights=None)
    backend.load_state_dict(weights)
    backend_name = "efficientnet"
    input_features = 1280

    num_classes = 1
    num_outputs = 2
    num_external_features = 2
    ### Parse ###
    config = {}
    config['backend'] = backend
    config['backend_name'] = backend_name
    config['input_features'] = input_features
    config['num_classes'] = num_classes
    config['num_external_features'] = num_external_features
    config['num_outputs'] = num_outputs
    return config

def config_rectum_efficientnet(weights) -> dict:
    ### Define Params ###
    backend = tv.models.efficientnet_v2_s(weights=None)
    backend.load_state_dict(weights)
    backend_name = "efficientnet"
    input_features = 1280

    num_classes = 1
    num_outputs = 2
    num_external_features = 4
    ### Parse ###
    config = {}
    config['backend'] = backend
    config['backend_name'] = backend_name
    config['input_features'] = input_features
    config['num_classes'] = num_classes
    config['num_external_features'] = num_external_features
    config['num_outputs'] = num_outputs
    return config


def config_prostate_resnet18(weights) -> dict:
    ### Define Params ###
    backend = tv.models.resnet18(weights=None)
    backend.load_state_dict(weights)
    backend_name = "resnet18"
    input_features = 512

    num_classes = 1
    num_outputs = 2
    num_external_features = 3
    ### Parse ###
    config = {}
    config['backend'] = backend
    config['backend_name'] = backend_name
    config['input_features'] = input_features
    config['num_classes'] = num_classes
    config['num_external_features'] = num_external_features
    config['num_outputs'] = num_outputs
    return config


def config_prostate_resnet50(weights) -> dict:
    ### Define Params ###
    backend = tv.models.resnet50(weights=None)
    backend.load_state_dict(weights)
    backend_name = "resnet50"
    input_features = 2048

    num_classes = 1
    num_outputs = 2
    num_external_features = 3
    ### Parse ###
    config = {}
    config['backend'] = backend
    config['backend_name'] = backend_name
    config['input_features'] = input_features
    config['num_classes'] = num_classes
    config['num_external_features'] = num_external_features
    config['num_outputs'] = num_outputs
    return config





def config_breast_efficientnet(weights) -> dict:
    ### Define Params ###
    backend = tv.models.efficientnet_v2_s(weights=None)
    backend.load_state_dict(weights)
    backend_name = "efficientnet"
    input_features = 1280

    num_classes = 1
    num_outputs = 4
    num_external_features = 16
    ### Parse ###
    config = {}
    config['backend'] = backend
    config['backend_name'] = backend_name
    config['input_features'] = input_features
    config['num_classes'] = num_classes
    config['num_external_features'] = num_external_features
    config['num_outputs'] = num_outputs
    return config




class MILNet(tc.nn.Module):
    def __init__(self, backend, backend_name, input_features, num_classes, num_external_features, num_outputs):
        super(MILNet, self).__init__()
        self.conv_layers = tc.nn.Sequential(*list(backend.children())[:-1])
        self.conv_layers.requires_grad_ = False
        self.input_features = input_features
        self.num_classes = num_classes
        self.backend_name = backend_name
        self.num_external_features = num_external_features

        if self.backend_name == "efficientnet":
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
        embeddings = self.conv_layers(x).view(-1, self.input_features)
        A = self.attention(embeddings)
        A = tc.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = tc.mm(A, embeddings)
        output = self.classifier(tc.cat((M, features), dim=1))
        return output



class MILNetExtended(tc.nn.Module):
    def __init__(self, backend, backend_name, input_features, num_classes, num_external_features, num_outputs):
        super(MILNetExtended, self).__init__()
        self.conv_layers = tc.nn.Sequential(*list(backend.children())[:-1])
        self.conv_layers.requires_grad_ = False
        self.input_features = input_features
        self.num_classes = num_classes
        self.backend_name = backend_name
        self.num_external_features = num_external_features

        if self.backend_name == "efficientnet":
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
            tc.nn.Linear(self.L, self.L),
            tc.nn.PReLU(),
            tc.nn.Linear(self.L, self.L),
            tc.nn.PReLU(),
        )
  
        self.attention = tc.nn.Sequential(
            tc.nn.Linear(self.L, self.D),
            tc.nn.Tanh(),
            tc.nn.Linear(self.D, self.K)
        )

        self.classifier = tc.nn.Sequential(
            tc.nn.Linear(self.L*self.K + self.num_external_features, num_outputs)
        )


    def forward(self, x, features):
        embeddings = self.conv_layers(x).view(-1, self.input_features)
        embeddings = self.intermediate_layer(embeddings)
        A = self.attention(embeddings)
        A = tc.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = tc.mm(A, embeddings)
        output = self.classifier(tc.cat((M, features), dim=1))
        return output



def test_channels_1():
    device = "cpu"
    config = config_prostate_resnet18()
    model = MILNet(**config).to(device)
    num_samples = 34
    num_channels = 3
    y_size = 224
    x_size = 224
    input = tc.randn((num_samples, num_channels, y_size, x_size), device=device)
    features = tc.zeros((1, 2), device=device)
    result = model(input, features)
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=(input, features), device=device, depth=5)  
    
def test_channels_2():
    device = "cpu"
    config = config_prostate_resnet50()
    model = MILNet(**config).to(device)
    num_samples = 34
    num_channels = 3
    y_size = 224
    x_size = 224
    input = tc.randn((num_samples, num_channels, y_size, x_size), device=device)
    features = tc.zeros((1, 2), device=device)
    result = model(input, features)
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=(input, features), device=device, depth=5)  


def test_channels_3():
    device = "cpu"
    weights_path = r'/home/mw/Projects/Chaimeleon_Championship/models/EffNetV2_S_StateDict'
    weights = tc.load(weights_path, map_location=tc.device('cpu'))
    config = config_breast_efficientnet(weights)
    model = MILNet(**config).to(device)
    num_samples = 34
    num_channels = 3
    y_size = 384
    x_size = 384
    input = tc.randn((num_samples, num_channels, y_size, x_size), device=device)
    features = tc.zeros((1, 16), device=device)
    result = model(input, features)
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=(input, features), device=device, depth=5) 



def run():
    # test_channels_2()
    test_channels_3()
    pass

if __name__ == "__main__":
    run()