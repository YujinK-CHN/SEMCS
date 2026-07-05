"""
SEMFO-inspired evolution for SESiL solver population.

Bidirectional selection with complementary skill preference,
permutation-aligned weight merging for neural networks.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def build_mating_scores(fitness_matrix, threshold=0.1, weight_extra=0.9, weight_common=0.1):
    """
    Build directional mating score matrix from fitness.

    Args:
        fitness_matrix: (M, K) array, higher = better (1/rank)
        threshold: minimum fitness to consider a task "known"
        weight_extra: weight for skills B has that A doesn't
        weight_common: weight for overlapping skills
    Returns:
        scores: dict of dict, scores[a][b] = how much a wants b
    """
    M = fitness_matrix.shape[0]
    scores = {}
    for a in range(M):
        scores[a] = {}
        fa = fitness_matrix[a]
        known_a = {i for i, v in enumerate(fa) if v > threshold}
        for b in range(M):
            if a == b:
                continue
            fb = fitness_matrix[b]
            known_b = {i for i, v in enumerate(fb) if v > threshold}
            extra = known_b - known_a
            common = known_a & known_b
            score = weight_extra * sum(fb[i] for i in extra)
            score += weight_common * sum(fb[i] for i in common)
            scores[a][b] = score
    return scores


def bidirectional_selection(scores, max_attempts=100):
    """
    SEMFO bidirectional selection: each agent probabilistically picks a partner,
    only mutual matches form pairs. Unmatched agents are loners.

    Returns:
        pairs: list of (a, b) tuples
        loners: list of unmatched indices
    """
    available = set(scores.keys())
    pairs = []

    for _ in range(max_attempts):
        if len(available) <= 1:
            break

        choices = {}
        for m in available:
            candidates = {k: v for k, v in scores[m].items() if k in available and k != m}
            if candidates:
                models = list(candidates.keys())
                weights = np.array([candidates[k] for k in models], dtype=float)
                if weights.sum() == 0:
                    choices[m] = np.random.choice(models)
                else:
                    probs = weights / weights.sum()
                    choices[m] = np.random.choice(models, p=probs)
            else:
                choices[m] = None

        matched = set()
        for a in list(available):
            b = choices.get(a)
            if b is not None and choices.get(b) == a:
                pair = tuple(sorted((a, b)))
                if pair not in pairs:
                    pairs.append(pair)
                matched.update([a, b])

        available -= matched
        if not matched:
            break

    loners = list(available)
    return pairs, loners


def compute_permutation(model_a, model_b, sample_obs, device):
    """
    Compute neuron permutation alignment between two models by matching
    hidden activations via Hungarian algorithm on correlation matrix.

    Args:
        model_a, model_b: nn.Module (the encoder Sequential)
        sample_obs: tensor of sample observations
    Returns:
        permutations: dict mapping layer_name -> permutation array
        layer_names: ordered list of Linear layer names
    """
    activations_a = {}
    activations_b = {}
    hooks = []
    layer_names = []

    for name, module in model_a.named_modules():
        if isinstance(module, nn.Linear) and name != '':
            layer_names.append(name)
            activations_a[name] = []
            def hook_fn(mod, inp, out, n=name, store=activations_a):
                store[n].append(out.detach())
            hooks.append(module.register_forward_hook(hook_fn))

    for name, module in model_b.named_modules():
        if isinstance(module, nn.Linear) and name != '':
            activations_b[name] = []
            def hook_fn(mod, inp, out, n=name, store=activations_b):
                store[n].append(out.detach())
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        sample = sample_obs[:min(256, len(sample_obs))].to(device)
        expected_dim = next(model_a.parameters()).shape[0]
        if sample.shape[-1] < expected_dim:
            pad = torch.zeros(sample.shape[0], expected_dim - sample.shape[-1],
                              dtype=sample.dtype, device=device)
            sample = torch.cat([sample, pad], dim=-1)
        elif sample.shape[-1] > expected_dim:
            sample = sample[:, :expected_dim]
        model_a(sample)
        model_b(sample)

    for h in hooks:
        h.remove()

    permutations = {}
    for name in layer_names[:-1]:
        if not activations_a.get(name) or not activations_b.get(name):
            continue
        act_a = activations_a[name][0]
        act_b = activations_b[name][0]
        if act_a.dim() > 2:
            act_a = act_a.reshape(-1, act_a.shape[-1])
            act_b = act_b.reshape(-1, act_b.shape[-1])

        n_neurons = act_a.shape[-1]
        corr = torch.zeros(n_neurons, n_neurons, device=device)
        for i in range(n_neurons):
            for j in range(n_neurons):
                a_i = act_a[:, i]
                b_j = act_b[:, j]
                a_c = a_i - a_i.mean()
                b_c = b_j - b_j.mean()
                corr[i, j] = (a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)

        cost = -corr.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros(n_neurons, dtype=int)
        perm[row_ind] = col_ind
        permutations[name] = perm

    return permutations, layer_names


def merge_state_dicts(sd_a, sd_b, permutations, layer_names):
    """
    Apply permutation to sd_b, then average with sd_a.

    Args:
        sd_a, sd_b: state dicts (will not be modified)
        permutations: dict from compute_permutation
        layer_names: ordered Linear layer names in the encoder
    Returns:
        merged_sd: averaged state dict
    """
    sd_b = {k: v.clone() for k, v in sd_b.items()}

    for name, perm in permutations.items():
        perm_tensor = torch.tensor(perm, dtype=torch.long)
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key in sd_b:
            sd_b[weight_key] = sd_b[weight_key][perm_tensor]
        if bias_key in sd_b:
            sd_b[bias_key] = sd_b[bias_key][perm_tensor]
        next_weight_key = None
        for ln in layer_names:
            if ln > name:
                next_weight_key = f"{ln}.weight"
                break
        if next_weight_key and next_weight_key in sd_b:
            sd_b[next_weight_key] = sd_b[next_weight_key][:, perm_tensor]

    merged_sd = {}
    for key in sd_a:
        if key in sd_b:
            merged_sd[key] = (sd_a[key] + sd_b[key]) / 2.0
        else:
            merged_sd[key] = sd_a[key].clone()
    return merged_sd


def _average_state_dicts(sd_a, sd_b):
    """Simple weight averaging of two state dicts."""
    merged = {}
    for key in sd_a:
        if key in sd_b:
            merged[key] = (sd_a[key] + sd_b[key]) / 2.0
        else:
            merged[key] = sd_a[key].clone()
    return merged


@torch.no_grad()
def merge_actors(actor_a, actor_b, sample_obs, device):
    """
    Merge two R_Actor models via weight averaging.

    Entity actor mode (IntegrationCritic): direct weight averaging.
    Flat actor mode (MLP encoder): permutation-aligned weight averaging.

    The act_layer and rnn are always averaged directly.
    Returns a new actor with merged weights (deep-copied from actor_a).
    """
    offspring = copy.deepcopy(actor_a)

    if actor_a.use_entity_actor:
        merged_sd = _average_state_dicts(
            actor_a.integration.state_dict(),
            actor_b.integration.state_dict())
        offspring.integration.load_state_dict(merged_sd)
    else:
        permutations, layer_names = compute_permutation(
            actor_a.encoder, actor_b.encoder, sample_obs, device)
        merged_encoder_sd = merge_state_dicts(
            actor_a.encoder.state_dict(),
            actor_b.encoder.state_dict(),
            permutations, layer_names)
        offspring.encoder.load_state_dict(merged_encoder_sd)

    merged_act_sd = _average_state_dicts(
        actor_a.act_layer.state_dict(),
        actor_b.act_layer.state_dict())
    offspring.act_layer.load_state_dict(merged_act_sd)

    if hasattr(actor_a, 'rnn'):
        merged_rnn_sd = _average_state_dicts(
            actor_a.rnn.state_dict(),
            actor_b.rnn.state_dict())
        offspring.rnn.load_state_dict(merged_rnn_sd)

    return offspring


@torch.no_grad()
def merge_critics(critic_a, critic_b):
    """Merge two R_Critic models via direct weight averaging."""
    offspring = copy.deepcopy(critic_a)
    merged_sd = _average_state_dicts(
        critic_a.state_dict(), critic_b.state_dict())
    offspring.load_state_dict(merged_sd)
    return offspring
