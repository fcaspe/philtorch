import torch
from torch import optim
import torch.nn.functional as F
from philtorch.lti import filtfilt, lfilter, lfilter_zi
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

def lowpass_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    Q: torch.Tensor,
):
    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q

    b0 = (1 - torch.cos(w0)) / 2
    b1 = 1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    b = torch.stack([b0, b1, b2], dim=-1) / a0
    a = torch.stack([a1, a2], dim=-1) / a0
    return b, a


sr = 16000
T = 6
target_cutoff = 1000.0
target_Q = 10.0
device = "cpu"
param2coefs = partial(lowpass_biquad_coef, sr)
x = torch.randn(1, sr * T, device=device)
b_target, a_target = param2coefs(
    torch.tensor(target_cutoff, device=device), torch.tensor(target_Q, device=device)
)
zi = lfilter_zi(a_target, b_target)
y, _ = lfilter(b_target, a_target, x, zi=zi * x[:, :1])

est_cutoff = torch.nn.Parameter(
    torch.tensor(200.0, device=device)
)
est_Q = torch.nn.Parameter(
    torch.tensor(0.707, device=device)
)
optimizer = optim.Adam([est_cutoff, est_Q], lr=0.2)

loss_history = []
cutoff_history = []
Q_history = []

with tqdm(range(20000)) as pbar:
    for step in pbar:
        b_est, a_est = param2coefs(est_cutoff, est_Q)
        zi = lfilter_zi(a_est, b_est)
        y_est, _ = lfilter(b_est, a_est, x, zi=zi * x[:, :1])
        loss = F.mse_loss(y_est, y)
        loss_history.append(loss.item())
        cutoff_history.append(est_cutoff.item())
        Q_history.append(est_Q.item())

        if loss.item() < 1e-6:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(
            f"loss: {loss.item():.6f}, cutoff: {est_cutoff.item():.2f}, Q: {est_Q.item():.2f}"
        )

fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

ax[0].plot(loss_history)
ax[0].set_title("Loss")
ax[0].set_xlabel("Iteration")
ax[0].set_xlim(1, len(loss_history))
ax[0].set_ylabel("MSE Loss")
ax[1].plot(cutoff_history)
ax[1].set_title("Estimated Cutoff Frequency")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Cutoff Frequency (Hz)")
ax[1].axhline(target_cutoff, color="red", linestyle="--", label="Target Cutoff")
ax[1].legend()
ax[2].plot(Q_history)
ax[2].set_title("Estimated Q Factor")
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Q Factor")
ax[2].axhline(target_Q, color="red", linestyle="--", label="Target Q")
ax[2].legend()
plt.tight_layout()
plt.savefig('test.png')
