import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMS:
    """Running mean/std for reward normalization."""
    def __init__(self, epsilon=1e-4, shape=(1,), device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M = torch.zeros(shape, device=self.device)
        self.S = torch.ones(shape, device=self.device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs / (self.n + bs)) / (self.n + bs)
        self.M = new_M
        self.S = new_S
        self.n += bs
        return self.M, self.S


def compute_apt_reward(source, target, knn_k=16, knn_avg=True,
                       knn_rms=True, knn_clip=0.0005, rms=None):
    """Compute particle-based entropy reward via k-NN distances in representation space.

    Args:
        source: (b1, d) representation tensor
        target: (b2, d) representation tensor
        knn_k: number of nearest neighbors
        knn_avg: average over all k neighbors (True) or use only k-th (False)
        knn_rms: normalize reward by running std
        knn_clip: clip threshold (negative to disable)
        rms: RMS instance for normalization (created if None)

    Returns:
        reward: (b1,) intrinsic reward tensor
        rms: the RMS instance used (pass back for persistence)
    """
    if rms is None:
        rms = RMS(device=source.device)

    b1, b2 = source.size(0), target.size(0)
    sim_matrix = torch.norm(
        source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1),
        dim=-1, p=2)
    reward, _ = sim_matrix.topk(knn_k, dim=1, largest=False, sorted=True)

    if not knn_avg:
        reward = reward[:, -1].reshape(-1, 1)
        if knn_rms:
            reward = reward / rms(reward)[1]
        if knn_clip >= 0.0:
            reward = torch.maximum(reward - knn_clip, torch.zeros_like(reward))
    else:
        reward = reward.reshape(-1, 1)
        if knn_rms:
            reward = reward / rms(reward)[1]
        if knn_clip >= 0.0:
            reward = torch.maximum(reward - knn_clip, torch.zeros_like(reward))
        reward = reward.reshape((b1, knn_k)).mean(dim=1)

    reward = torch.log(reward + 1.0)
    return reward, rms
