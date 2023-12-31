import torch
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import ppgs


###############################################################################
# Aggregate batch metric state
###############################################################################


class Metrics:

    def __init__(self, include_figures=False):
        self.metrics = [
            Accuracy(),
            CategoricalAccuracy(),
            JensenShannon(),
            TopKAccuracy(3),
            Loss()]
        if include_figures:
            self.metrics.append(DistanceMatrix())

    def __call__(self):
        results = {}
        for metric in self.metrics:
            results.update(metric())
        return results

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(
        self, predicted_logits, target_indices):
        for metric in self.metrics:
            metric.update(predicted_logits, target_indices)


###############################################################################
# Batch metrics
###############################################################################


class Accuracy:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {f'Accuracy': (self.true_positives / self.count).item()}

    def reset(self):
        self.count = 0
        self.true_positives = 0

    def update(self, predicted_logits, target_indices):
        # Predicted category is the maximum logit
        predicted_indices = predicted_logits.argmax(dim=1)

        # Compare to target indices
        self.true_positives += torch.logical_and(
            predicted_indices == target_indices,
            target_indices != -100
        ).sum()

        # Update count
        self.count += (target_indices != -100).sum()


class CategoricalAccuracy:

    def __init__(self):
        self.reset()
        self.map = {i: phoneme for i, phoneme in enumerate(ppgs.PHONEMES)}

    def __call__(self):
        if self.totals is not None:
            assert self.totals.shape == self.counts.shape
        else:
            return None
        output = {}
        for i in range(0, self.totals.shape[0]):
            output[f'Accuracy/{self.map[i]}'] = (
                self.totals[i] / self.counts[i]).item()
            output[f'Total/{self.map[i]}'] = self.totals[i].item()
            output[f'Count/{self.map[i]}'] = self.counts[i].item()
        return output

    def reset(self):
        self.totals = None
        self.counts = None

    def update(self, predicted_logits, target_indices):
        """Update per-category accuracy"""
        # Unroll time dimension
        # batch x classes x time -> batch * time x classes
        predicted_logits = torch.transpose(
            predicted_logits, 1, 2).flatten(0, 1)
        # batch x time -> batch * time
        target_indices = target_indices.flatten()

        # Remove padding
        keep_indices = (target_indices != -100)
        target_indices = target_indices[keep_indices]
        predicted_logits = predicted_logits[keep_indices]

        # Get predicted category
        predicted_indices = predicted_logits.argmax(dim=1)
        predicted_onehots = torch.nn.functional.one_hot(
            predicted_indices,
            num_classes=predicted_logits.shape[-1])

        # Get target category
        target_onehots = torch.nn.functional.one_hot(
            target_indices,
            num_classes=predicted_logits.shape[-1])

        # Update totals
        marginal_totals = torch.mul(
            predicted_onehots,
            target_onehots).sum(dim=0)
        if self.totals is None:
            self.totals = marginal_totals
        else:
            self.totals += marginal_totals

        # Update counts
        marginal_counts = target_onehots.sum(dim=0)
        if self.counts is None:
            self.counts = marginal_counts
        else:
            self.counts += marginal_counts


class JensenShannon:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {f'JSD': (self.total / self.count).item()}

    def reset(self):
        self.total = torch.tensor(0., dtype=torch.float32)
        self.count = 0

    def update(self, predicted_logits, target_indices):
        """Update the total JSD"""
        # Unroll time dimension
        # batch x classes x time -> batch * time x classes
        predicted_logits = torch.transpose(
            predicted_logits, 1, 2).flatten(0, 1)
        # batch x time -> batch * time
        target_indices = target_indices.flatten()

        # Remove padding
        keep_indices = (target_indices != -100)
        target_indices = target_indices[keep_indices]
        predicted_logits = predicted_logits[keep_indices]

        # Convert to probabilities
        predicted_probs = torch.nn.functional.softmax(
            predicted_logits,
            dim=-1
        ).to(torch.float)
        target_probs = torch.nn.functional.one_hot(
            target_indices,
            num_classes=predicted_logits.shape[-1]
        ).to(torch.float)

        # Compute pronunciation distance
        jsd = ppgs.distance(
            predicted_probs.T,
            target_probs.T,
            reduction='sum')

        # Update total and count
        self.total += jsd.item()
        self.count += (target_indices != -100).sum()


class Loss:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {f'loss': (self.total / self.count).item()}

    def reset(self):
        self.total = 0.
        self.count = 0

    def update(self, predicted_logits, target_indices):
        """Update the total cross entropy loss"""
        self.total += ppgs.loss(
            predicted_logits,
            target_indices,
            reduction='sum'
        ).item()
        self.count += (target_indices != -100).sum()


class TopKAccuracy:

    def __init__(self, k):
        self.k = k
        self.reset()

    def __call__(self):
        accuracy = (self.correct_in_top_k / self.count).item()
        return {f'Top-{self.k} Accuracy/': accuracy}

    def reset(self):
        self.count = 0
        self.correct_in_top_k = 0

    def update(self, predicted_logits, target_indices):
        # Unroll time dimension
        # batch x classes x time -> batch * time x classes
        predicted_logits = torch.transpose(
            predicted_logits, 1, 2).flatten(0, 1)
        # batch x time -> batch * time
        target_indices = target_indices.flatten()

        # Remove padding
        nonpad_indices = target_indices != -100
        predicted_logits = predicted_logits[nonpad_indices]
        target_indices = target_indices[nonpad_indices]

        # Compute accuracy
        top_k = torch.topk(predicted_logits, self.k, dim=-1)
        top_k_indices = top_k.indices
        self.correct_in_top_k += (
            top_k_indices == target_indices[:, None]).sum()
        self.count += len(target_indices)


###############################################################################
# Batch-updating figures
###############################################################################


class ConfusionMatrix:

    def __init__(self):
        self.reset()

    def _normalized(self):
        return (self.matrix / self.matrix.sum(dim=1)).cpu()

    def _render(self):
        figure = plt.figure(dpi=400, figsize=(6, 6))
        ax = figure.add_subplot()
        ax.matshow(self._normalized())
        phones = ppgs.PHONEMES
        num_phones = len(ppgs.PHONEMES)
        ax.locator_params('both', nbins=num_phones)
        ax.set_xticklabels([''] + phones, rotation='vertical')
        ax.set_yticklabels([''] + phones)
        ax.set_ylabel('Ground Truth Phoneme')
        ax.set_xlabel('Model Predicted Probabilities')
        figure.align_labels()
        return figure

    def __call__(self):
        return {f'ConfusionMatrix': self._render()}

    def reset(self):
        self.matrix = None

    def update(self, predicted_logits, target_indices):
        # Unroll time dimension
        # batch x classes x time -> batch * time x classes
        predicted_logits = torch.transpose(
            predicted_logits, 1, 2).flatten(0, 1)
        # batch x time -> batch * time
        target_indices = target_indices.flatten()

        # Normalize
        predicted_probs = torch.nn.functional.softmax(predicted_logits, dim=1)

        # Remove padding
        nonpad_indices = target_indices != -100
        predicted_probs = predicted_probs[nonpad_indices]
        target_indices = target_indices[nonpad_indices]

        # Maybe initialize matrix
        if self.matrix is None:
            self.matrix = torch.zeros(
                (predicted_probs.shape[-1], predicted_probs.shape[-1])
            ).to(predicted_logits.device)

        # Update
        self.matrix[target_indices] += predicted_probs


class DistanceMatrix:

    def __init__(self, weighted=True):
        if weighted:
            self.weights = ppgs.load.phoneme_weights()
        else:
            self.weights = torch.ones(ppgs.OUTPUT_CHANNELS)
        self.reset()

    def _normalized(self):
        return (self.matrix / self.matrix.sum(dim=1)[:, None]).cpu()

    def _render(self):
        figure = plt.figure(dpi=400, figsize=(6, 6))
        ax = figure.add_subplot()
        mat = ax.matshow(self._normalized(), norm=PowerNorm(gamma=1/3))
        phones = ppgs.PHONEMES
        num_phones = len(ppgs.PHONEMES)
        ax.locator_params('both', nbins=num_phones)
        ax.set_xticklabels([''] + phones, rotation='vertical')
        ax.set_yticklabels([''] + phones)
        ax.tick_params(
            axis='x',
            top=True,
            bottom=True,
            labelbottom=True,
            labeltop=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        ax.figure.colorbar(mat, cax=cax)
        figure.align_labels()

        phone_pairs = [
            ('f', 'v'),
            ('s', 'z'),
            ('sh', 'zh')]

        padding = 0.5 + 0.2

        for phone0, phone1 in phone_pairs:
            idx0 = ppgs.PHONEMES.index(phone0)
            idx1 = ppgs.PHONEMES.index(phone1)
            indices = [idx0, idx1]
            for center in [indices, list(reversed(indices))]:
                box_start_coords = (center[0]-padding, center[1]-padding)
                box_width = padding*2
                box_height = padding*2
                ax.add_patch(
                    plt.Rectangle(
                        box_start_coords,
                        box_width,
                        box_height,
                        facecolor='none',
                        edgecolor='red',
                        linewidth=0.5
                    )
                )

        return figure

    def __call__(self):
        return {f'DistanceMatrix': self._render()}

    def reset(self):
        self.matrix = None
        self.count = None

    def update(self, predicted_logits, target_indices):
        predicted_logits = predicted_logits.to(torch.float32)

        # Unroll time dimension
        # batch x classes x time -> batch * time x classes
        predicted_logits = torch.transpose(
            predicted_logits, 1, 2).flatten(0, 1)
        # batch x time -> batch * time
        target_indices = target_indices.flatten()

        # Normalize
        predicted_probs = torch.nn.functional.softmax(predicted_logits, dim=1)

        # Apply phoneme class weighting
        predicted_probs = torch.mul(
            predicted_probs,
            self.weights[None, :].to(predicted_probs.device))

        # Get maximal phoneme
        predicted_indices = predicted_probs.argmax(dim=1)

        # Remove padding
        nonpad_indices = target_indices != -100
        predicted_probs = predicted_probs[nonpad_indices]
        predicted_indices = predicted_indices[nonpad_indices]
        target_indices = target_indices[nonpad_indices]

        # Maybe initialize matrix
        if self.matrix is None:
            self.matrix = torch.zeros(
                (predicted_probs.shape[-1], predicted_probs.shape[-1])
            ).to(device=predicted_logits.device, dtype=predicted_logits.dtype)

        # Maybe initialize counts
        if self.count is None:
            self.count = torch.zeros(
                predicted_probs.shape[-1]
            ).to(predicted_logits.device, dtype=torch.long)

        # Update matrix
        self.matrix.scatter_add_(
            0,
            predicted_indices[:, None].expand(-1, predicted_probs.shape[-1]),
            predicted_probs)

        # Update counts
        self.count += target_indices.bincount()
