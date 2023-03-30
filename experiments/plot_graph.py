import json
import matplotlib.pyplot as plt
import numpy as np
import os

render_vdo_path = r"../result/ppo/test"
fpath = os.path.join(render_vdo_path, "summary.json")
with open(fpath, "r") as f:
    data = json.load(f)
print(data.keys())

n_episode = data["n/ep"]
rwds = np.array(data["rews"])
max_cycles = data["len"]
noise_mu = np.array(data["noises_mu"])
noise_sig = np.array(data["noises_sig"])
num_agents = rwds.shape[1]

if noise_mu.ndim != 4:
    exit()

num_noise_type = noise_mu.shape[-1]

x = np.arange(max_cycles)
print(noise_mu.shape)

for ep in range(n_episode):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(f"episode {ep + 1}")
    for t in range(num_noise_type):
        # horizontal align
        # ax_mu = fig.add_subplot(num_noise_type, 2, t + 1, title=f"mu-{t}", xlabel="step")
        # ax_sig = fig.add_subplot(num_noise_type, 2, t + 2, title=f"sig-{t}", xlabel="step")

        # vertical align
        ax_mu = fig.add_subplot(2, num_noise_type, t + 1, title=f"mu-{t}", xlabel="step", xlim=(0, max_cycles))
        ax_sig = fig.add_subplot(2, num_noise_type, t + 1 + num_noise_type, title=f"sig-{t}", xlabel="step", xlim=(0, max_cycles))
        # ax_mu.set_xlabel
        for i in range(num_agents):
            ax_mu.plot(x, noise_mu[:, ep, i, t], label=f"P{i}", marker='.')
            ax_mu.plot(x, np.zeros_like(x), color="black")
            ax_sig.plot(x, noise_sig[:, ep, i, t], label=f"P{i}", marker='.')
            ax_sig.plot(x, np.zeros_like(x), color="black")

        ax_mu.legend(loc=1, fontsize=8)
        ax_sig.legend(loc=1, fontsize=8)
    
    fig.tight_layout()
    # fig.savefig(os.path.join(render_vdo_path, f"render-{ep + 1:02d}", "noise.png"), dpi=200)
plt.show()
