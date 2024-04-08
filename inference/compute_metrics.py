import yaml
import time
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validation import validate
from data_preprocessing import (
    TrainDataPreprocessor,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter


def compute_metrics(input_dim, context_dim, gpu, load_kwargs, data_kwargs, base_kwargs):
    # Device
    if gpu != None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
    # Set up the log directories
    if load_kwargs["log_name"] is not None:
        save_dir = "../src/checkpoints/%s" % load_kwargs["log_name"]
        log_dir_test = "./logs/%s_%s" % (
            load_kwargs["log_name"],
            load_kwargs["process"],
        )
        save_dir_test = "./checkpoints/%s_%s" % (
            load_kwargs["log_name"],
            load_kwargs["process"],
        )
        if not os.path.exists(log_dir_test):
            os.makedirs(log_dir_test)
        if not os.path.exists(save_dir_test):
            os.makedirs(save_dir_test)

    else:
        raise ValueError("No log name specified!")

    # Initialize the tensorboard writer
    writer = SummaryWriter(log_dir=log_dir_test)
    # Load the models
    if load_kwargs["cfm"] is True:
        from create_cfm_model import resume_cfm_model

        model, _, _ = resume_cfm_model(save_dir, load_kwargs["checkpoint"], device)

    elif load_kwargs["dnf"] is True:
        from modded_basic_nflows import load_mixture_model

        model, _, _, _, _, _ = load_mixture_model(
            "",  # ! device, useless
            model_dir=save_dir,
            filename=load_kwargs["checkpoint"],
        )
    else:
        raise ValueError("No model specified!")

    # Load the data
    if input_dim == 5:
        dataset = TrainDataPreprocessor(data_kwargs)
        X_dataset, Y_dataset = dataset.get_dataset()
    elif input_dim > 15:
        dataset = TrainDataPreprocessor(data_kwargs)
        X_dataset, Y_dataset = dataset.get_dataset()
    else:
        raise ValueError("Input dim not supported")

    # At this point the data is already preprocessed
    X_dataset = torch.tensor(X_dataset).float()
    Y_dataset = torch.tensor(Y_dataset).float()

    # Simulate the data
    batch_size = data_kwargs["batch_size"]
    total_batches = len(X_dataset) // batch_size

    model.to(device)
    model.eval()

    if load_kwargs["cfm"] is True:
        from modded_cfm import ModelWrapper

        sigma = base_kwargs["cfm"]["sigma"]
        timesteps = base_kwargs["cfm"]["timesteps"]
        sampler = ModelWrapper(model, context_dim)
        if base_kwargs["cfm"]["ode_backend"] == "torchdyn":
            from torchdyn.models import NeuralODE

            node = NeuralODE(
                sampler,
                solver="dopri5",
                sensitivity="adjoint",
                atol=1e-5,
                rtol=1e-5,
            )
        t_span = torch.linspace(0, 1, timesteps).to(device)

    samples_list = []
    with torch.no_grad():
        loss = torch.zeros(1, device=device)
        log_p = torch.zeros(1, device=device)
        log_det = torch.zeros(1, device=device)
        cfm_loss = torch.zeros(1, device=device)

        for i in tqdm(range(0, len(X_dataset), batch_size), ascii=True):
            y = Y_dataset[i : i + batch_size].to(device)
            x = X_dataset[i : i + batch_size].to(device)

            if load_kwargs["cfm"] is True:
                x0 = torch.randn(len(y), x.shape[1]).to(device)
                initial_conditions = torch.cat([x0, y], dim=-1)
                if base_kwargs["cfm"]["ode_backend"] == "torchdyn":
                    samples = node.trajectory(initial_conditions, t_span)[
                        timesteps - 1, :, : X_dataset.shape[1]
                    ]
                elif base_kwargs["cfm"]["ode_backend"] == "torchdiffeq":
                    from torchdiffeq import odeint

                    samples = odeint(
                        sampler,
                        initial_conditions,
                        t_span,
                        method="dopri5",
                        atol=1e-5,
                        rtol=1e-5,
                    )[timesteps - 1, :, : X_dataset.shape[1]]
                else:
                    raise ValueError("ODE backend not supported")

                # Compute the loss

            elif load_kwargs["dnf"] is True:
                # Compute the loss
                tmp_log_p, tmp_log_det = model(inputs=x, context=y)
                tmp_loss = -torch.mean(tmp_log_p + tmp_log_det)

                loss += tmp_loss.item()
                log_p += torch.mean(-tmp_log_p).item()
                log_det += torch.mean(-tmp_log_det).item()

                samples = model.sample(1, context=y)
            else:
                raise ValueError("No model specified!")

            samples_list.append(samples.detach().cpu().numpy())

    loss /= total_batches
    log_p /= total_batches
    log_det /= total_batches

    writer.add_scalar("Losses/total_loss", loss, 0)
    writer.add_scalar("Losses/log_p", log_p, 0)
    writer.add_scalar("Losses/log_det", log_det, 0)

    samples = np.concatenate(samples_list, axis=0)
    samples = np.array(samples).reshape(-1, X_dataset.shape[1])

    X_dataset = X_dataset.cpu().numpy()
    Y_dataset = Y_dataset.cpu().numpy()

    # Invert the preprocessing
    if data_kwargs["standardize"] == True:
        X_dataset = dataset.scaler_x.inverse_transform(X_dataset)
        if data_kwargs["flavour_ohe"]:
            Y_dataset[:, 0:5] = dataset.scaler_y.inverse_transform(Y_dataset[:, 0:5])
        else:
            Y_dataset = dataset.scaler_y.inverse_transform(Y_dataset)
        samples = dataset.scaler_x.inverse_transform(samples)

    if data_kwargs["flavour_ohe"]:
        if Y_dataset.shape[1] > 6:
            tmp = Y_dataset[:, 5:]
            b = tmp[:, 5]
            c = tmp[:, 4]
            flavour = np.zeros(len(b))
            flavour[np.where(b == 1)] = 5
            flavour[np.where(c == 1)] = 4

        Y_dataset = np.hstack(
            (Y_dataset[:, :4], flavour.reshape(-1, 1), Y_dataset[:, 6].reshape(-1, 1))
        )

    if data_kwargs["physics_scaling"] == True:
        X_dataset, Y_dataset = dataset.invert_physics_scaling(X_dataset, Y_dataset)
        samples, _ = dataset.invert_physics_scaling(samples, Y_dataset)

    X_dataset[:, 4] = np.round(X_dataset[:, 4])
    if input_dim > 15:
        X_dataset[:, 11] = np.round(X_dataset[:, 11])
        X_dataset[:, 12] = np.round(X_dataset[:, 12])
        X_dataset[:, 14] = np.round(X_dataset[:, 14])

    samples[:, 4] = np.rint(samples[:, 4])
    if input_dim > 15:
        samples[:, 11] = np.rint(samples[:, 11])  # ncharged
        samples[:, 12] = np.rint(samples[:, 12])  # nneutral
        samples[:, 14] = np.rint(samples[:, 14])  # nSV
    # clip to physical values based on the boundaries of X_test_cpu
    for i in range(samples.shape[1]):
        samples[:, i] = np.clip(
            samples[:, i],
            X_dataset[:, i].min(),
            X_dataset[:, i].max(),
        )

    Y_dataset[:, 4] = np.rint(Y_dataset[:, 4])

    np.save(os.path.join(save_dir_test, "samples.npy"), samples)
    np.save(os.path.join(save_dir_test, "X_test_cpu.npy"), X_dataset)
    np.save(os.path.join(save_dir_test, "Y_test_cpu.npy"), Y_dataset)

    # Validation
    if input_dim == 5:
        validate(samples, X_dataset, Y_dataset, save_dir_test, 0, writer)
    elif input_dim > 15:
        validate(samples, X_dataset, Y_dataset, save_dir_test, 0, writer)
    else:
        raise ValueError("Input dim not supported")


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        config_path = "./configs/" + args[1]
    else:
        config_path = "./configs/config.yaml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    input_dim = config["input_dim"]
    context_dim = config["context_dim"]
    gpu = config["gpu"]
    load_kwargs = config["load_kwargs"]
    data_kwargs = config["data_kwargs"]
    base_kwargs = config["base_kwargs"]

    compute_metrics(input_dim, context_dim, gpu, load_kwargs, data_kwargs, base_kwargs)
