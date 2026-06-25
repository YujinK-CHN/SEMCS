import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from base_policy.utils.util import init_, check


class VAEEncoder(nn.Module):
    def __init__(self, args, input_dim, skill_dim, device):
        super(VAEEncoder, self).__init__()
        self.args = args
        self.skill_dim = skill_dim
        self.hidden_size = args.hidden_size

        self.encoder = nn.Sequential(
            init_(args, nn.Linear(input_dim, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(args, nn.Linear(self.hidden_size, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(args, nn.Linear(self.hidden_size, 2 * skill_dim))
        )

        self.decoder = nn.Sequential(
            init_(args, nn.Linear(skill_dim, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(args, nn.Linear(self.hidden_size, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(args, nn.Linear(self.hidden_size, input_dim))
        )

        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x, is_training=False):
        param = self.encoder(x)
        mu, log_sigma = torch.split(param, [self.skill_dim, self.skill_dim], dim=-1)
        sigma = torch.exp(log_sigma)

        epsilon = torch.randn_like(mu)
        z = mu + epsilon * sigma

        train_info = {}
        if is_training:
            priors = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
            posteriors = torch.distributions.Normal(mu, sigma)
            loss_kl = self.args.coef_kl_loss * torch.sum(
                torch.distributions.kl.kl_divergence(posteriors, priors), dim=-1)

            x_re = self.decoder(z)
            loss_re = torch.sum((x_re - x) ** 2, dim=-1).mean()

            train_info["vae_loss_kl"] = loss_kl
            train_info["vae_loss_re"] = loss_re

        return z, train_info


class EncoderPopulation(nn.Module):
    def __init__(self, args, input_dim, skill_dim, device):
        super(EncoderPopulation, self).__init__()
        self.args = args
        self.M = args.num_encoders
        self.skill_dim = skill_dim
        self.device = device

        self.encoders = nn.ModuleList([
            VAEEncoder(args, input_dim, skill_dim, device)
            for _ in range(self.M)
        ])

        self.fitness_history = defaultdict(lambda: defaultdict(list))

    def forward(self, entity_ob_list, n_agents, n_entities, is_training=False):
        skill_list = []
        all_kl = []
        all_re = []

        for task_idx, (entity_ob, na) in enumerate(zip(entity_ob_list, n_agents)):
            bs_na = entity_ob.shape[0]

            if self.args.sesil_use_entity_obs:
                obs_input = entity_ob
            else:
                obs_input = entity_ob.mean(dim=-2)

            encoder_ids = torch.arange(na, device=self.device) % self.M
            encoder_ids = encoder_ids.unsqueeze(0).expand(bs_na // na, -1).reshape(-1)

            z_out = torch.zeros(bs_na, self.skill_dim, device=self.device)
            task_kl = torch.tensor(0.0, device=self.device)
            task_re = torch.tensor(0.0, device=self.device)
            count = 0

            for enc_id in range(self.M):
                mask = encoder_ids == enc_id
                if not mask.any():
                    continue

                if self.args.sesil_use_entity_obs:
                    enc_input = obs_input[mask]
                    enc_input = enc_input.mean(dim=-2)
                else:
                    enc_input = obs_input[mask]

                z, info = self.encoders[enc_id](enc_input, is_training)
                z_out[mask] = z

                if is_training and info:
                    task_kl = task_kl + info["vae_loss_kl"].mean()
                    task_re = task_re + info["vae_loss_re"]
                    count += 1

            skill_list.append(z_out)
            if is_training and count > 0:
                all_kl.append(task_kl / count)
                all_re.append(task_re / count)

        train_info = {}
        if is_training:
            train_info["vae_loss_kl"] = all_kl if all_kl else [torch.tensor(0.0, device=self.device)]
            train_info["vae_loss_re"] = all_re if all_re else [torch.tensor(0.0, device=self.device)]

        return skill_list, train_info

    def record_fitness(self, encoder_id, task_idx, episode_return):
        self.fitness_history[encoder_id][task_idx].append(episode_return)

    def get_encoder_id(self, agent_idx):
        return agent_idx % self.M

    @torch.no_grad()
    def evolve(self, obs_batch):
        if not self.fitness_history:
            return {"pairs": 0, "loners": 0}

        task_ids = set()
        for enc_id in self.fitness_history:
            task_ids.update(self.fitness_history[enc_id].keys())
        task_ids = sorted(task_ids)
        K = len(task_ids)

        if K == 0:
            return {"pairs": 0, "loners": 0}

        factorial_cost = np.full((self.M, K), np.inf)
        for enc_id in range(self.M):
            for k_idx, task_id in enumerate(task_ids):
                returns = self.fitness_history[enc_id].get(task_id, [])
                if returns:
                    factorial_cost[enc_id, k_idx] = -np.mean(returns)

        has_data = np.isfinite(factorial_cost)
        if not has_data.any():
            self.fitness_history.clear()
            return {"pairs": 0, "loners": 0}

        factorial_cost[~has_data] = np.nanmax(factorial_cost[has_data]) + 1.0

        pairs, loners = self._bidirectional_selection(factorial_cost)

        for (a, b) in pairs:
            self._merge_encoders(a, b, obs_batch)

        for lone in loners:
            self._mutate_encoder(lone)

        self.fitness_history.clear()

        return {"pairs": len(pairs), "loners": len(loners)}

    def _bidirectional_selection(self, factorial_cost):
        M, K = factorial_cost.shape
        ranks = np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1
        fitness_matrix = 1.0 / ranks

        scores = {}
        threshold = self.args.evo_threshold
        w_extra = self.args.evo_weight_extra
        w_common = self.args.evo_weight_common

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
                score = w_extra * sum(fb[i] for i in extra)
                score += w_common * sum(fb[i] for i in common)
                scores[a][b] = score

        available = set(range(M))
        pairs = []
        for _ in range(100):
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

    def _merge_encoders(self, idx_a, idx_b, obs_batch):
        enc_a = self.encoders[idx_a]
        enc_b = self.encoders[idx_b]

        activations_a = {}
        activations_b = {}

        hooks_a = []
        hooks_b = []
        layer_names = []

        for name, module in enc_a.encoder.named_modules():
            if isinstance(module, nn.Linear) and name != '':
                layer_names.append(name)
                activations_a[name] = []
                def hook_a(mod, inp, out, n=name):
                    activations_a[n].append(out.detach())
                hooks_a.append(module.register_forward_hook(hook_a))

        for name, module in enc_b.encoder.named_modules():
            if isinstance(module, nn.Linear) and name != '':
                activations_b[name] = []
                def hook_b(mod, inp, out, n=name):
                    activations_b[n].append(out.detach())
                hooks_b.append(module.register_forward_hook(hook_b))

        if obs_batch is not None and len(obs_batch) > 0:
            sample_obs = obs_batch[0]
            if self.args.sesil_use_entity_obs:
                sample_obs = sample_obs.mean(dim=-2)
            sample_obs = sample_obs[:min(256, sample_obs.shape[0])]
            enc_a.encoder(sample_obs)
            enc_b.encoder(sample_obs)

        for h in hooks_a + hooks_b:
            h.remove()

        sd_a = enc_a.state_dict()
        sd_b = enc_b.state_dict()

        permutations = {}
        for name in layer_names[:-1]:
            if name not in activations_a or name not in activations_b:
                continue
            if not activations_a[name] or not activations_b[name]:
                continue

            act_a = activations_a[name][0]
            act_b = activations_b[name][0]

            if act_a.dim() > 2:
                act_a = act_a.reshape(-1, act_a.shape[-1])
                act_b = act_b.reshape(-1, act_b.shape[-1])

            n_neurons = act_a.shape[-1]
            corr = torch.zeros(n_neurons, n_neurons, device=self.device)
            for i in range(n_neurons):
                for j in range(n_neurons):
                    a_i = act_a[:, i]
                    b_j = act_b[:, j]
                    a_i_centered = a_i - a_i.mean()
                    b_j_centered = b_j - b_j.mean()
                    denom = (a_i_centered.norm() * b_j_centered.norm() + 1e-8)
                    corr[i, j] = (a_i_centered * b_j_centered).sum() / denom

            cost = -corr.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            perm = np.zeros(n_neurons, dtype=int)
            perm[row_ind] = col_ind
            permutations[name] = perm

        for key in sd_a:
            if key in sd_b:
                sd_a[key] = (sd_a[key] + sd_b[key]) / 2.0

        enc_a.load_state_dict(sd_a)
        enc_b.load_state_dict(sd_a)

    def _mutate_encoder(self, idx):
        enc = self.encoders[idx]
        std = self.args.evo_mutation_std
        for param in enc.parameters():
            param.data.add_(torch.randn_like(param.data) * std)
