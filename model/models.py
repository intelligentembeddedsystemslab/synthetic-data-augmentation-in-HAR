import os

import torch
import torch.nn as nn
import math

from utils.utils import paint, makedir
from utils.utils_pytorch import (
    get_info_params,
    get_info_layers,
    init_weights_orthogonal,
)
from utils.utils_attention import SelfAttention, TemporalAttention
from settings import get_args

__all__ = ["create"]

"""
Implementation based on https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
"""

## Define our DeepConvLSTM class, subclassing nn.Module.
class DeepConvLSTM(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        n_head,
        d_feedforward,
        train_mode,
        experiment,
        cuda
    ):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))

        self.dropout = nn.Dropout(0.5) #set like this in paper
        self.lstm = nn.LSTM(input_dim * filter_num, hidden_dim, num_layers=2) #number of layers according to paper

        self.classifier = nn.Linear(hidden_dim, num_class)

        self.activation = nn.ReLU()

        self.model = model
        self.dataset = dataset
        self.experiment = experiment

        #create buffer for center-loss contraint
        if torch.cuda.is_available():
            self.register_buffer(
                "centers", (torch.randn(num_class, hidden_dim).to(device=cuda))
            )
        else: 
            self.register_buffer(
                "centers", (torch.randn(num_class, hidden_dim))
            )

        makedir(self.path_checkpoints)
        makedir(self.path_logs)
        makedir(self.path_visuals)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.dropout(x)
        x, h = self.lstm(x)
        x = x[-1, :, :]


        #return features for center-loss:
        z = x.div(
            torch.norm(x, p=2, dim=1, keepdim=True).expand_as(x)
        )

        out = self.classifier(x)

        return z, out

    @property
    def path_checkpoints(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/checkpoints/"

    @property
    def path_logs(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/logs/"

    @property
    def path_visuals(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/visuals/"

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        activation,
        sa_div,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            filter_num * input_dim,
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )
        

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        outputs, h = self.rnn(x)

        # apply temporal attention on GRU outputs
        out = self.ta(outputs)
        return out


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)


class AttendDiscriminate(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        n_head,
        d_feedforward,
        train_mode,
        experiment,
        cuda
    ):
        super(AttendDiscriminate, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
        )

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(hidden_dim, num_class)
        if torch.cuda.is_available():
            self.register_buffer(
                "centers", (torch.randn(num_class, self.hidden_dim).to(device=cuda))
            )
        else: 
            self.register_buffer(
                "centers", (torch.randn(num_class, self.hidden_dim))
            )

        if train_mode:
            makedir(self.path_checkpoints)
            makedir(self.path_logs)
        makedir(self.path_visuals)

    def forward(self, x):
        feature = self.fe(x)
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        out = self.dropout(feature)
        logits = self.classifier(out)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/checkpoints/"

    @property
    def path_logs(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/logs/"

    @property
    def path_visuals(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/visuals/"


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(LearnedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.positional_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        positional_encoding = self.positional_embeddings(positions)
        return x + positional_encoding


class TransformHAR(nn.Module):
    """
    Trabsformer encoder configuration from https://github.com/yolish/har-with-imu-transformer
    """

    def __init__(
        self,
        model,
        dataset,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        n_head,
        d_feedforward,
        train_mode,
        experiment,
        cuda
    ):
        super(TransformHAR, self).__init__()
        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.dataset = dataset

        
        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout_cls)

        self.positional_encoder = PositionalEncoding(
            d_model=filter_num * input_dim, dropout=dropout_rnn, vocab_size=1280
        )
        #self.positional_encoder = LearnedPositionalEncoding(filter_num * input_dim, 1280)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=filter_num * input_dim,
            nhead=n_head, #8
            dim_feedforward=d_feedforward, #128
            dropout=dropout,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=enc_num_layers, #3
            norm = nn.LayerNorm(filter_num * input_dim)
        )

        self.cls_token = nn.Parameter(torch.zeros((1, filter_num * input_dim)), requires_grad=True)

        self.classifier = nn.Linear(filter_num * input_dim, num_class)

        #create buffer for center-loss contraint
        if torch.cuda.is_available():
            self.register_buffer(
                "centers", (torch.randn(num_class, filter_num * input_dim).to(device=cuda))
            )
        else: 
            self.register_buffer(
                "centers", (torch.randn(num_class, filter_num * input_dim))
            )

        makedir(self.path_checkpoints)
        makedir(self.path_logs)
        makedir(self.path_visuals)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        x = x4

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.dropout(x)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([cls_token, x])

        x = self.positional_encoder(x)

        x = self.transformer_encoder(x)[0]

        #return features for center-loss:
        z = x.div(
            torch.norm(x, p=2, dim=1, keepdim=True).expand_as(x)
        )

        out = self.classifier(x)
        return z, out
    
    @property
    def path_checkpoints(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/checkpoints/"

    @property
    def path_logs(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/logs/"

    @property
    def path_visuals(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/visuals/"



__factory = {
    "AttendDiscriminate": AttendDiscriminate,
    "DeepConvLSTM": DeepConvLSTM,
    "TransformHAR" : TransformHAR,
}


def create(model, config):
    if model not in __factory.keys():
        raise KeyError(f"[!] Unknown HAR model: {model}")

    return __factory[model](**config)
