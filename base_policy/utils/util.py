import numpy as np
import math
import torch

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    if module.weight is not None and module.weight.data.numel() > 0:
        weight_init(module.weight.data, gain=gain)
    if module.bias is not None and module.bias.data.numel() > 0:
        bias_init(module.bias.data)
    return module

def init_(args, m):
    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
    return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)


def init_gru(args, gru: nn.GRU):
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            if args.use_orthogonal:
                nn.init.orthogonal_(param)
            else:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            # zero all biases first
            nn.init.constant_(param, 0.0)
    return gru

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def check(inputs, device):
    '''
    input is numpy/tensor or a list of numpy/tensor
    device is a dict consist of dtype and device of torch.device
    output is a list of tensor
    '''
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        output = [torch.from_numpy(input).to(**device) if type(input) == np.ndarray else input.to(**device) for input in inputs]
    else:
        output = torch.from_numpy(inputs).to(**device) if type(inputs) == np.ndarray else inputs.to(**device)
    return output


def check_(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def negative_log_loss(e):
    return -e

def cross_entropy_loss(logits, targets):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    return loss_fn(logits, targets)

def kl_loss(log_probs, probs, reduction="none"):
    kl_loss = F.kl_div(input=log_probs, target=probs, reduction=reduction)
    return kl_loss

def mse_loss(e):
    return e**2/2

def pearson_loss(x, y, eps=1e-8):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    numerator = (x * y).sum(dim=1)
    denominator = x.norm(dim=1) * y.norm(dim=1) + eps

    return -(numerator / denominator).mean()


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def deep_copy_data(data):
    return [[np.copy(arr) for arr in step] for step in data]


def stack_over_steps_with_padding(data, window_size, pad_value=-1):
    """
    Args:
        data: list of list of numpy arrays with shape (x, y, z)
        window_size: number of consecutive steps to stack (n)
        pad_value: value to use when padding missing steps at the end

    Returns:
        result: list of list of arrays with shape (x, y, n, z)
    """
    data = deep_copy_data(data)
    num_steps = len(data)
    num_tasks = len(data[0])
    result = []

    for start in range(num_steps):
        step_group = []
        for task in range(num_tasks):
            task_data = []
            max_z = 0
            # Collect data or placeholders for the window
            for offset in range(window_size):
                idx = start + offset
                if idx < num_steps:
                    arr = data[idx][task]
                else:
                    # pad shape will match the first element in the task
                    arr = np.full_like(data[start][task], pad_value)
                task_data.append(arr)
                max_z = max(max_z, arr.shape[2])

            # Pad z-axis to max_z for consistency
            padded = [np.pad(t, ((0,0), (0,0), (0, max_z - t.shape[2])), constant_values=pad_value)
                      for t in task_data]
            stacked = np.stack(padded, axis=2)  # shape: (x, y, n, z)
            step_group.append(stacked)
        result.append(step_group)
    return result



def print_numpy_array_sizes(**arrays):
    total = 0
    print("=== Array Memory Usage ===")
    for name, arr in arrays.items():
        if isinstance(arr, np.ndarray):
            size_mb = arr.nbytes / 1024**2
            total += size_mb
            print(f"{name:<40}: {size_mb:.2f} MB | shape={arr.shape} dtype={arr.dtype}")
        else:
            print(f"{name:<40}: Not a NumPy array (type={type(arr)})")
    print(f"\nTotal memory usage: {total:.2f} MB")


def print_cuda(tag, is_training=True):
    if is_training:
        print(f"[{tag}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def report_model_memory(model, name="model"):
    total_mem = 0
    print(f"\n{name} parameters memory usage (in MB):")
    for n, p in model.named_parameters():
        if p.device.type != "cuda":
            continue
        mem = p.element_size() * p.nelement() / (1024 ** 2)
        total_mem += mem
        # print(f"  {n:<40} {str(list(p.shape)):>20}  --> {mem:7.2f} MB")
    print(f"Total for {name}: {total_mem:.2f} MB")