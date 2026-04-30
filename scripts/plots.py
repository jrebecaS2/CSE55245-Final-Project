import matplotlib.pyplot as plt
import numpy as np


# generates plots comparing evaluation benchmark scores

def create_figure(figsize=(10, 6)):
    plt.style.use('seaborn-v0_8-poster')
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def finalize_plot(ax, title, xlabel, ylabel, xticks=None, xticklabels=None, legend=True, save_path=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if legend:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# exploration 2

def plot_beta_sweep(beta_values, ifeval_scores, gsm8k_scores, mbpp_scores, save_path=None):
    fig, ax = create_figure()

    bar_width = 0.2
    x = np.arange(len(beta_values))

    ax.bar(x - bar_width, ifeval_scores, width=bar_width, label='IFEval', color='#a6cee3')
    ax.bar(x, gsm8k_scores, width=bar_width, label='GSM8K',  color='#b2df8a')
    ax.bar(x + bar_width, mbpp_scores, width=bar_width, label='MBPP',  color='#fdbf6f')

    finalize_plot(
        ax,
        title="β Sweep Results",
        xlabel="β",
        ylabel="Score",
        xticks=x,
        xticklabels=[str(b) for b in beta_values],
        save_path=save_path
    )


# exploration 1

def plot_filter_comparison(models, gsm8k_scores, mbpp_scores, save_path=None):
    fig, ax = create_figure()

    bar_width = 0.3
    x = np.arange(len(models))

    ax.bar(x - bar_width / 2, gsm8k_scores, width=bar_width, label='GSM8K', color='#b2df8a')
    ax.bar(x + bar_width / 2, mbpp_scores, width=bar_width, label='MBPP', color='#fdbf6f')

    finalize_plot(
        ax,
        title="Filtered vs Unfiltered DPO Performance",
        xlabel="Model",
        ylabel="Score",
        xticks=x,
        xticklabels=models,
        save_path=save_path
    )


if __name__ == "__main__":
    beta_values = [0.01, 0.1, 0.5]
    ifeval_scores = [0.3375, 0.339, 0.35]
    gsm8k_scores = [0.06, 0.05, 0.05]
    mbpp_scores = [0.12, 0.1, 0.08]

    plot_beta_sweep(beta_values, ifeval_scores, gsm8k_scores, mbpp_scores)

    # Filter comparison example
    models = ["Unfiltered DPO", "Filtered DPO"]
    gsm8k = [0.05, 0.05]  # replace
    mbpp = [0.10, 0.10]   # replace

    plot_filter_comparison(models, gsm8k, mbpp)