import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# render_vdo_path = r"../result/ppo/test"
folder = input("Please enter the folder name under 'result/ppo': ") or "test"
render_vdo_path = r"../result/ppo/" + folder

fpath = os.path.join(render_vdo_path, "summary.json")
with open(fpath, "r") as f:
    data = json.load(f)
print(data.keys())

def plot_graph(data, path, has_noise):
    n_episode = data["n/ep"]
    rwds = np.array(data["rews"])
    max_cycles = int(data["len"])
    noise_mu = np.array(data["noises_mu"])
    noise_sig = np.array(data["noises_sig"])
    rwds_detail = np.array(data["rews_detail"])
    num_agents = rwds.shape[1]

    fpath_toggle = os.path.join(render_vdo_path, "toggle.log")
    toggle_eps = []

    with open(fpath_toggle, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if line.startswith("[") and line.endswith("]"):
                toggle = line[1:-1].split()
                toggle = list(map(int, toggle))
                toggle_eps.append(toggle)
        
        toggle_eps = np.array(toggle_eps)

    toggle_eps = toggle_eps.reshape((n_episode, max_cycles, num_agents))

    def plot_toggle(ep):
        df = pd.DataFrame(toggle_eps[ep], columns=np.arange(num_agents))
        # df.plot.bar(subplots=True, xlabel="step", ylabel="mask", legend=False, title=[f"episode {ep + 1}/agent {i}" for i in range(num_agents)], xticks=np.arange(max_cycles, step=5), figsize=(8, 6))
        df.plot.bar(subplots=True, xlabel="step", ylabel="mask", legend=False, title=[f"agent {i}" for i in range(num_agents)], xticks=np.arange(max_cycles, step=5), figsize=(8, 6), yticks=[])
        plt.tight_layout()
        # plt.show()
        img_path = os.path.join(path, f"render-{ep + 1:02d}", "toggle.png")
        plt.savefig(img_path, dpi=200)
        plt.close()
        print(f"reward graph generated to {img_path}")
    
    for ep in range(n_episode):
        plot_toggle(ep)

plot_graph(data, render_vdo_path, False)