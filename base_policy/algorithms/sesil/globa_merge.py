"""
GLOBA-based merging for SEBAL.

Adapted from the GLOBA (GLObal Basis Analysis) codebase for MARL policy merging.
Operates on state_dict pairs with a shared base model (common pretrain).
"""
import numpy as np
import torch


def _svd_truncate(delta, energy_threshold):
    """SVD truncation by energy threshold. Returns truncated delta, U[:,:k], V[:,:k]."""
    U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
    V = Vt.T
    total_energy = torch.sum(S ** 2)
    if total_energy <= 0:
        return delta, U[:, :1], V[:, :1]
    k = max(1, (torch.cumsum(S ** 2, dim=0) / total_energy <= energy_threshold).sum().item())
    delta_truncated = U[:, :k] @ torch.diag(S[:k]) @ V[:, :k].T
    return delta_truncated, U[:, :k], V[:, :k]


def _build_global_basis(U1, U2, V1, V2, energy_threshold=0.999):
    """Build global orthonormal bases Pu, Pv from concatenated singular vectors."""
    U_cat = torch.cat((U1, U2), dim=1)
    V_cat = torch.cat((V1, V2), dim=1)

    try:
        U_cat_d = U_cat.double()
        Pu_d, Du, _ = torch.linalg.svd(U_cat_d, full_matrices=False)
        total_u = torch.sum(Du ** 2)
        k_u = torch.searchsorted(torch.cumsum(Du ** 2, dim=0) / total_u, energy_threshold).item() + 1 if total_u > 0 else 1
        Pu = Pu_d[:, :k_u].to(U_cat.dtype)

        V_cat_d = V_cat.double()
        Pv_d, Dv, _ = torch.linalg.svd(V_cat_d, full_matrices=False)
        total_v = torch.sum(Dv ** 2)
        k_v = torch.searchsorted(torch.cumsum(Dv ** 2, dim=0) / total_v, energy_threshold).item() + 1 if total_v > 0 else 1
        Pv = Pv_d[:, :k_v].to(V_cat.dtype)
    except torch._C._LinAlgError:
        return None, None

    return Pu, Pv


def _energy_prune(C, energy_threshold):
    """Prune C matrix by keeping top entries covering target energy."""
    if C.numel() == 0:
        return torch.zeros_like(C)
    energy = C ** 2
    total = torch.sum(energy)
    if total.item() == 0.0 or energy_threshold <= 0.0:
        return torch.zeros_like(C)
    if energy_threshold >= 1.0:
        return C.clone()

    sorted_e, sorted_idx = torch.sort(energy.flatten(), descending=True)
    cumulative = torch.cumsum(sorted_e, dim=0)
    n_keep = min(torch.searchsorted(cumulative, total * energy_threshold, right=False).item() + 1, C.numel())
    mask = torch.zeros(C.numel(), dtype=torch.bool, device=C.device)
    mask[sorted_idx[:n_keep]] = True
    return torch.where(mask.view_as(C), C, torch.zeros_like(C))


def _classify_and_merge(C1_masked, C2_masked, scales):
    """
    Classify C2 entries into 6 types relative to C1 and recombine.

    scales: dict with keys 'c1', 'D_minus', 'D_plus', 'E', 'B', 'C', 'A'
    """
    C_merged = torch.zeros_like(C1_masked)

    if scales['c1'] != 0:
        C_merged += C1_masked * scales['c1']

    C1_nz = C1_masked != 0
    C2_nz = C2_masked != 0

    C1_rows = C1_nz.any(dim=1)
    C1_cols = C1_nz.any(dim=0)

    if scales['D_minus'] != 0:
        mask = C1_nz & C2_nz & (torch.sign(C1_masked) != torch.sign(C2_masked))
        C_merged += (C2_masked * mask) * scales['D_minus']

    if scales['D_plus'] != 0:
        mask = C1_nz & C2_nz & (torch.sign(C1_masked) == torch.sign(C2_masked))
        C_merged += (C2_masked * mask) * scales['D_plus']

    if scales['E'] != 0:
        mask = (C1_masked == 0) & C1_rows.unsqueeze(1) & C1_cols
        C_merged += (C2_masked * mask) * scales['E']

    if scales['B'] != 0:
        mask = C1_rows.unsqueeze(1) & ~C1_cols
        C_merged += (C2_masked * mask) * scales['B']

    if scales['C'] != 0:
        mask = ~C1_rows.unsqueeze(1) & C1_cols
        C_merged += (C2_masked * mask) * scales['C']

    if scales['A'] != 0:
        mask = ~C1_rows.unsqueeze(1) & ~C1_cols
        C_merged += (C2_masked * mask) * scales['A']

    return C_merged


def _compute_type_energies(C1_masked, C2_masked):
    """
    Compute energy in each of the 6 types of C2 relative to C1.
    Returns dict with keys 'D_minus', 'D_plus', 'E', 'B', 'C', 'A'.
    """
    C1_nz = C1_masked != 0
    C2_nz = C2_masked != 0
    C1_rows = C1_nz.any(dim=1)
    C1_cols = C1_nz.any(dim=0)

    energies = {}

    mask_Dm = C1_nz & C2_nz & (torch.sign(C1_masked) != torch.sign(C2_masked))
    energies['D_minus'] = torch.sum((C2_masked * mask_Dm) ** 2).item()

    mask_Dp = C1_nz & C2_nz & (torch.sign(C1_masked) == torch.sign(C2_masked))
    energies['D_plus'] = torch.sum((C2_masked * mask_Dp) ** 2).item()

    mask_E = (C1_masked == 0) & C1_rows.unsqueeze(1) & C1_cols
    energies['E'] = torch.sum((C2_masked * mask_E) ** 2).item()

    mask_B = C1_rows.unsqueeze(1) & ~C1_cols
    energies['B'] = torch.sum((C2_masked * mask_B) ** 2).item()

    mask_C = ~C1_rows.unsqueeze(1) & C1_cols
    energies['C'] = torch.sum((C2_masked * mask_C) ** 2).item()

    mask_A = ~C1_rows.unsqueeze(1) & ~C1_cols
    energies['A'] = torch.sum((C2_masked * mask_A) ** 2).item()

    return energies


def _is_skip_layer(key):
    """Check if a parameter should skip SVD (bias, norm layers)."""
    return key.endswith('.bias') or 'norm' in key.lower() or 'layernorm' in key.lower()


@torch.no_grad()
def globa_merge_state_dicts(base_sd, sd_a, sd_b, args):
    """
    Merge two state dicts using GLOBA relative to a shared base.

    Args:
        base_sd: base model state dict (common pretrain)
        sd_a, sd_b: finetuned state dicts (solvers)
        args: namespace with globa_* hyperparameters
    Returns:
        merged_sd: merged state dict (absolute weights, not deltas)
    """
    scales = {
        'c1': 1.0,
        'D_minus': args.globa_scale_D_minus,
        'D_plus': args.globa_scale_D_plus,
        'E': args.globa_scale_E,
        'B': args.globa_scale_B,
        'C': args.globa_scale_C,
        'A': args.globa_scale_A,
    }

    merged_sd = {}

    for key in sd_a:
        if key not in base_sd or key not in sd_b:
            merged_sd[key] = sd_a[key].clone()
            continue

        delta_a = sd_a[key] - base_sd[key]
        delta_b = sd_b[key] - base_sd[key]

        if _is_skip_layer(key) or delta_a.dim() < 2:
            merged_sd[key] = base_sd[key] + delta_a * args.globa_skip_weight_1 + delta_b * args.globa_skip_weight_2
            continue

        delta_a_t, U1, V1 = _svd_truncate(delta_a, args.globa_svd_energy_1)
        delta_b_t, U2, V2 = _svd_truncate(delta_b, args.globa_svd_energy_2)

        Pu, Pv = _build_global_basis(U1, U2, V1, V2, args.globa_global_basis_energy)
        if Pu is None:
            merged_sd[key] = base_sd[key] + delta_a * args.globa_skip_weight_1 + delta_b * args.globa_skip_weight_2
            continue

        C1 = Pu.T @ delta_a_t @ Pv
        C2 = Pu.T @ delta_b_t @ Pv

        C1_masked = _energy_prune(C1, args.globa_c_prune_energy_1)
        C2_masked = _energy_prune(C2, args.globa_c_prune_energy_2)

        C_merged = _classify_and_merge(C1_masked, C2_masked, scales)
        delta_merged = Pu @ C_merged @ Pv.T

        merged_sd[key] = base_sd[key] + delta_merged

    return merged_sd


@torch.no_grad()
def globa_mating_scores(base_sd, solver_sds, args):
    """
    Compute directional mating preference scores for all solver pairs
    using GLOBA type decomposition on weight space.

    Args:
        base_sd: base model state dict
        solver_sds: list of solver state dicts
        args: namespace with globa_* and globa_mate_* hyperparameters
    Returns:
        scores: dict of dict, scores[i][j] = how much solver i wants solver j
    """
    M = len(solver_sds)
    keys_to_analyze = [k for k in base_sd if k in solver_sds[0] and not _is_skip_layer(k) and base_sd[k].dim() >= 2]

    # Precompute per-solver SVD truncations and projections for efficiency
    solver_deltas = []
    for si in range(M):
        deltas = {}
        for key in keys_to_analyze:
            deltas[key] = solver_sds[si][key] - base_sd[key]
        solver_deltas.append(deltas)

    scores = {}
    for i in range(M):
        scores[i] = {}
        for j in range(M):
            if i == j:
                continue

            total_energies = {'D_minus': 0.0, 'D_plus': 0.0, 'E': 0.0, 'B': 0.0, 'C': 0.0, 'A': 0.0}

            for key in keys_to_analyze:
                delta_i = solver_deltas[i][key]
                delta_j = solver_deltas[j][key]

                if torch.sum(delta_i ** 2) == 0 or torch.sum(delta_j ** 2) == 0:
                    continue

                delta_i_t, U1, V1 = _svd_truncate(delta_i, args.globa_svd_energy_1)
                delta_j_t, U2, V2 = _svd_truncate(delta_j, args.globa_svd_energy_2)

                Pu, Pv = _build_global_basis(U1, U2, V1, V2, args.globa_global_basis_energy)
                if Pu is None:
                    continue

                C_i = Pu.T @ delta_i_t @ Pv
                C_j = Pu.T @ delta_j_t @ Pv

                C_i_m = _energy_prune(C_i, args.globa_c_prune_energy_1)
                C_j_m = _energy_prune(C_j, args.globa_c_prune_energy_2)

                layer_energies = _compute_type_energies(C_i_m, C_j_m)
                for t in total_energies:
                    total_energies[t] += layer_energies[t]

            score = (args.globa_mate_w_E * total_energies['E']
                     + args.globa_mate_w_BC * (total_energies['B'] + total_energies['C'])
                     + args.globa_mate_w_A * total_energies['A']
                     + args.globa_mate_w_Dp * total_energies['D_plus']
                     + args.globa_mate_w_Dm * total_energies['D_minus'])
            scores[i][j] = max(score, 0.0)

    return scores
