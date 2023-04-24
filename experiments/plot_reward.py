import numpy as np
import matplotlib.pyplot as plt

r_mul1 = [0, 1, 2, 3, 0.7, -0.2]
r_mul2 = [0, 1, 2, 5, 3, 3]
r_mul3 = [0, 1, 2, 5, 4, 3]
r_mul4 = [0, 1, 2, 7, 5, 3]
r_muls = [r_mul1, r_mul2, r_mul3, r_mul4]
x = np.arange(len(r_mul4))

def plot_single():
    for i, r_mul in enumerate(r_muls):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, np.zeros_like(x), "b")
        bars = ax.bar(x, r_mul)
        ax.bar_label(bars)
        plt.xlabel(r"#predators catching the same prey")
        plt.ylabel("reward multiplier")
        plt.title(f"multiplier: {tuple(r_mul)}")

        plt.savefig(f"images/multiplier_{i + 1}.png", dpi=200)
        # plt.show()
    plt.xlabel(r"#predators catching the same prey")
    plt.ylabel("reward multiplier")
    plt.title(f"multiplier: {tuple(r_mul)}")
    plt.savefig(f"images/multiplier_{i + 1}.png", dpi=200)

def plot_all():
    romans = ["(i)", "(ii)", "(iii)", "(iv)"]
    fig = plt.figure(figsize=(8, 6))
    for i, r_mul in enumerate(r_muls):
        title = f"{romans[i]} {tuple(r_mul)}"
        ax = fig.add_subplot(2, 2, i + 1)
        ax.plot(x, np.zeros_like(x), "b")
        bars = ax.bar(x, r_mul)
        ax.bar_label(bars)
        ax.set_xlabel(r"#predators")
        ax.set_ylabel("multiplier")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(f"images/multiplier_all.png", dpi=200)
    # plt.show()

def plot_sum():
    ratio32=[6.1, 19, 19, 25]
    ratio41=[1.8, 13, 17, 21]
    ratio50=[-2.5, 15, 15, 15]

    barWidth = 0.25
    fig = plt.subplots(figsize=(8, 6))
    
    # Set position of bar on X axis
    br1 = np.arange(len(ratio32))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    plt.bar(br1, ratio32, color="r", width=barWidth, edgecolor="grey", label="3:2")
    plt.bar(br2, ratio41, color="g", width=barWidth, edgecolor="grey", label="4:1")
    plt.bar(br3, ratio50, color="b", width=barWidth, edgecolor="grey", label="5:0") 
    
    plt.xlabel("candidates for reward multiplier")
    plt.ylabel("sum of majority's reward")
    plt.xticks([r + barWidth for r in range(len(r_muls))], r_muls)
    
    plt.legend()
    plt.savefig("images/multiplier_candidates.png", dpi=200)
    # plt.show()

plot_all()