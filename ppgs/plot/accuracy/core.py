import json

import matplotlib.pyplot as plt

import ppgs


###############################################################################
# Constants
###############################################################################


COLORS = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'lime']


###############################################################################
# Plot accuracy
###############################################################################


def from_eval(
    output_file,
    datasets=ppgs.DATASETS,
    representations=ppgs.ALL_REPRESENTATIONS
):
    """Plot accuracy of evaluated datasets and representations"""
    # Get evaluation files
    files = {
        r: ppgs.EVAL_DIR / r / 'overall.json' for r in representations}

    # Load results
    accuracies = {dataset: {} for dataset in datasets}
    for representation, file in files.items():
        with open(file, 'r') as file:
            results = json.load(file)
            for dataset in datasets:
                accuracies[dataset][representation] = \
                    results[dataset]['Accuracy']

    # Get average accuracy over datasets
    average = {}
    for representation in representations:
        average[representation] = \
            sum([
                accuracies[dataset][representation] for dataset in datasets
            ]) / len(datasets)

    # Get sort order
    representations = [
        representation for representation, _ in
        sorted(average.items(), key=lambda item: item[1], reverse=True)]

    # Display names
    representation_map = {
        'bottleneck': 'ASR bottleneck',
        'encodec': 'EnCodec',
        'mel': 'Mel spectrogram',
        'w2v2fb': 'Wav2vec 2.0',
        'w2v2fc': 'Charsiu'}
    dataset_map = {
        'commonvoice': 'Common Voice',
        'timit': 'TIMIT',
        'arctic': 'Arctic'}

    # Setup plot
    figure, axes = plt.subplots(
        1,
        4,
        sharey=True,
        figsize=(8, 2.8),
        width_ratios=[1, 1, 1, 1.4])
    inter_figure_distance = 0.075 -0.0025
    for i, dataset in enumerate(datasets):

        # Setup subplot
        ax = axes[i]
        ax.set_ylim(0.3, 0.9)
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.set_xticks(
            range(len(representations)),
            [representation_map[r] for r in representations],
            rotation=45,
            rotation_mode='anchor',
            ha='right',
            visible=False)
        ax.set_title(dataset_map[dataset])

        # Plot accuracies
        bar = ax.bar(
            range(len(representations)),
            [
                accuracies[dataset][representation]
                for representation in representations
            ],
            align='center',
            color=COLORS)

        # Plot gridlines
        if i == 0:
            xmin = 0
            xmax = 1 + inter_figure_distance
        elif i == len(datasets) - 1:
            xmin = 0 - inter_figure_distance
            xmax = 1
        else:
            xmin = 0 - inter_figure_distance
            xmax = 1 + inter_figure_distance
        for y in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ax.axhline(
                y,
                linestyle='dashed',
                clip_on=False,
                xmin=xmin,
                xmax=xmax)

    # Add legend
    lax = axes[-1]
    lax.axis('off')
    legend_labels = [
        representation_map[r] + f'\n(avg={average[r]:.3f})'
        for r in representations]
    top_legend = lax.legend(
        bar,
        legend_labels,
        loc='center',
        title=r'$\bf{Input\ representation:}$',
        frameon=False,
        fontsize=11
    )
    plt.subplots_adjust(wspace=0.05)

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0)
