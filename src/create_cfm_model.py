from typing import Tuple, Callable, Union, List
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs.
    imported from nflows https://github.com/bayesiains/nflows """

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class DenseResidualNet(nn.Module):
    """
    Modified from dingo fmpe https://github.com/dingo-gw/dingo/tree/FMPE
    A nn.Module consisting of a sequence of dense residual blocks. This is
    used to embed high dimensional input to a compressed output. Linear
    resizing layers are used for resizing the input and output to match the
    first and last hidden dimension, respectively.

    Module specs
    --------
        input dimension:    (batch_size, input_dim)
        output dimension:   (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple,
        activation: Callable = F.elu,
        dropout: float = 0.0,
        batch_norm: bool = True,
        context_features: int = None,
        time_varying: bool = False,
    ):
        """
        Parameters
        ----------
        input_dim : int
            dimension of the input to this module
        output_dim : int
            output dimension of this module
        hidden_dims : tuple
            tuple with dimensions of hidden layers of this module
        activation: callable
            activation function used in residual blocks
        dropout: float
            dropout probability for residual blocks used for reqularization
        batch_norm: bool
            flag that specifies whether to use batch normalization
        context_features: int
            Number of additional context features, which are provided to the residual
            blocks via gated linear units. If None, no additional context expected.
        """

        super(DenseResidualNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_res_blocks = len(self.hidden_dims)
        self.context_features = context_features
        self.time_varying = time_varying
        self.softmax = nn.Softmax(dim=0)

        prev_size = self.input_dim + context_features if context_features else self.input_dim
        if time_varying:
            prev_size += 1  # Add one for the time component
            self.context_features += 1
        self.initial_layer = nn.Linear(prev_size, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=self.hidden_dims[n],
                    context_features=self.context_features,
                    activation=activation,
                    dropout_probability=dropout,
                    use_batch_norm=batch_norm,
                )
                for n in range(self.num_res_blocks)
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dims[n - 1], self.hidden_dims[n])
                if self.hidden_dims[n - 1] != self.hidden_dims[n]
                else nn.Identity()
                for n in range(1, self.num_res_blocks)
            ]
            + [nn.Linear(self.hidden_dims[-1], self.output_dim)]
        )

    def forward(self, inputs, context=None, flow_time=None, weights=None):
        if weights is not None:
            # weights = self.softmax(weights)
            weights /= weights.max()
            # print(weights)
            inputs = inputs * weights
        if context is not None:
            inputs = torch.cat((inputs, context), dim=1)
        if self.time_varying:  # and flow_time is not None:
            inputs = torch.cat((inputs, flow_time), dim=1)
            new_context = torch.cat((context, flow_time), dim=1)
        else:
            new_context = context
        x = self.initial_layer(inputs)
        for block, resize_layer in zip(self.blocks, self.resize_layers):
            x = block(x, context=new_context)
            x = resize_layer(x)
        return x


class MergedMLP(nn.Module):
    """
    a merged MLP for the context and the time varying component
    """

    def __init__(
        self,
        in_shape: int,
        out_shape: int,
        context_features: int,
        hidden_sizes: list,
        activation=F.relu,
        activate_output: bool = False,
        batch_norm: bool = False,
        dropout_probability: float = 0.0,
        time_varying: bool = False,
    ):
        super().__init__()
        self.time_varying = time_varying

        layers = []
        prev_size = in_shape + context_features if context_features else in_shape
        if time_varying:
            prev_size += 1  # Add one for the time component

        # Initialize hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(p=dropout_probability))
            layers.append(activation)
            prev_size = size

        # Final layer
        layers.append(nn.Linear(prev_size, out_shape))
        if activate_output:
            layers.append(activation)

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor = None,
        flow_time: torch.Tensor = None,
    ):
        if context is not None:
            inputs = torch.cat((inputs, context), dim=1)
        if self.time_varying:  # and flow_time is not None:
            inputs = torch.cat((inputs, flow_time), dim=1)
        return self.network(inputs)


class GLUMergedMLP(nn.Module):
    def __init__(
        self,
        in_shape: int,
        out_shape: int,
        context_features: int,
        hidden_sizes: list,
        activation=F.relu,
        activate_output: bool = False,
        batch_norm: bool = False,
        dropout_probability: float = 0.0,
        time_varying: bool = False,
    ):
        super().__init__()
        self.time_varying = time_varying
        if time_varying:
            in_shape += 1

        self.initial_layer = nn.Linear(in_shape + context_features, hidden_sizes[0])
        if activation is not None:
            self.initial_activation = activation

        self.hidden_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()

        for i, (in_size, out_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            # print(in_size, out_size)
            self.hidden_layers.append(nn.Linear(in_size, out_size))
            if i % 2 == 0:
                self.glu_layers.append(nn.Linear(context_features, out_size))

        self.final_layer = nn.Linear(hidden_sizes[-1], out_shape)

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor = None,
        flow_time: torch.Tensor = None,
    ):
        if self.time_varying:  # and flow_time is not None:
            inputs = torch.cat((inputs, flow_time), dim=1)
        x = torch.cat((inputs, context), dim=1)

        # Initial layer
        x = self.initial_layer(x)
        x = self.initial_activation(x)
        # print(x.shape)

        # Hidden layers
        glu_index = 0
        for i, hidden in enumerate(self.hidden_layers):
            x = hidden(x)
            # print(i, x.shape)
            if i % 2 == 0:
                glu_out = self.glu_layers[glu_index](context)
                x = F.glu(torch.cat((x, glu_out), dim=1), dim=1)
                glu_index += 1
                # print(x.shape, glu_out.shape)
            else:
                x = self.initial_activation(x)

        # Final layer
        x = self.final_layer(x)

        return x


class AttentionMLP(nn.Module):
    def __init__(
        self,
        in_shape: int,
        out_shape: int,
        context_features: int,
        hidden_sizes: list,
        activation=F.relu,
        activate_output: bool = False,
        batch_norm: bool = False,
        dropout_probability: float = 0.0,
        time_varying: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        self.time_varying = time_varying
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(in_shape + context_features, num_heads)
        layers = []
        prev_size = in_shape + context_features  # Adding context_features for attention
        if time_varying:
            prev_size += 1  # Add one for the time component

        self.middle_layer = nn.Sequential(
            nn.Linear(
                16, 11
            ),  # Assuming inputs and attention_output have the same size
            nn.ReLU(),
        )

        # Initialize hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(p=dropout_probability))
            layers.append(activation)
            prev_size = size

        # Final layer
        layers.append(nn.Linear(prev_size, out_shape))
        if activate_output:
            layers.append(activation)

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor = None,
        flow_time: torch.Tensor = None,
    ):
        if context is not None:
            # Concatenating inputs and context for attention
            attention_input = torch.cat((inputs, context), dim=-1).unsqueeze(0)

            # Applying attention; assuming input shape is (batch_size, feature_dim)
            attention_output, _ = self.attention(
                attention_input, attention_input, attention_input
            )

            # Remove the extra dimension and concatenate attention output with original input
            inputs = torch.cat((inputs, attention_output.squeeze(0)), dim=-1)
            inputs = self.middle_layer(inputs)
        if self.time_varying:
            inputs = torch.cat((inputs, flow_time), dim=1)

        return self.network(inputs)


def build_cfm_model(input_dim, context_dim, base_kwargs):
    if base_kwargs["cfm"]["type"] == 'glu':
        model = GLUMergedMLP(
            in_shape=input_dim,
            out_shape=input_dim,
            context_features=context_dim,
            hidden_sizes=[
                base_kwargs["cfm"]["mlp_hidden_dim"]
                for _ in range(base_kwargs["cfm"]["mlp_num_hidden"])
            ],
            activation=nn.ReLU(),
            activate_output=False,
            batch_norm=base_kwargs["cfm"]["mlp_batch_norm"],
            time_varying=True,
        )
    elif base_kwargs["cfm"]["type"] == 'mlp':
        if base_kwargs["cfm"]["mlp_activation"] == "relu":
            activation = nn.ReLU()
        elif base_kwargs["cfm"]["mlp_activation"] == "elu":
            activation = nn.ELU()
        elif base_kwargs["cfm"]["mlp_activation"] == "gelu":
            activation = nn.GELU()
        else:
            raise NotImplementedError
        model = MergedMLP(
            in_shape=input_dim,
            out_shape=input_dim,
            context_features=context_dim,
            hidden_sizes=[
                base_kwargs["cfm"]["mlp_hidden_dim"]
                for _ in range(base_kwargs["cfm"]["mlp_num_hidden"])
            ],
            activation=activation,
            activate_output=False,
            batch_norm=base_kwargs["cfm"]["mlp_batch_norm"],
            time_varying=True,
        )
    elif base_kwargs["cfm"]["type"] == 'resnet':
        if base_kwargs["cfm"]["mlp_activation"] == "relu":
            resnet_activation = F.ReLU
        elif base_kwargs["cfm"]["mlp_activation"] == "elu":
            resnet_activation = F.elu
        elif base_kwargs["cfm"]["mlp_activation"] == "gelu":
            resnet_activation = F.gelu
        else:
            raise NotImplementedError
        model = DenseResidualNet(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dims=base_kwargs["cfm"]["mlp_hidden_dim"],
            activation=resnet_activation,
            dropout=base_kwargs["cfm"]["mlp_dropout"],
            batch_norm=base_kwargs["cfm"]["mlp_batch_norm"],
            context_features=context_dim,
            time_varying=True,
        )
    else:
        raise NotImplementedError

    return model


def save_cfm_model(model, epoch, lr, name, input_dim, context_dim, base_kwargs, path):
    if path is None:
        raise NameError("Model directory must be specified.")

    filename = name + f"_@epoch_{epoch}.pt"
    resume_filename = "checkpoint-latest.pt"

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    dict = {
        "epoch": epoch,
        "lr": lr,
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "context_dim": context_dim,
        "base_kwargs": base_kwargs,
    }

    torch.save(dict, p / filename)
    torch.save(dict, p / resume_filename)


def resume_cfm_model(path, filename):
    p = Path(path)
    dict = torch.load(p / filename)

    model = build_cfm_model(
        input_dim=dict["input_dim"],
        context_dim=dict["context_dim"],
        base_kwargs=dict["base_kwargs"],
    )
    model.load_state_dict(dict["model_state_dict"])

    return model, dict["epoch"], dict["lr"]
