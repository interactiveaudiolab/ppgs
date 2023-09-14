import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import ppgs

###############################################################################
# Aggregate batch metric state
###############################################################################


class Metrics:

    def __init__(self, display_suffix, include_figures=False):
        if include_figures:
            self.metrics = [
                Accuracy(display_suffix),
                CategoricalAccuracy(display_suffix),
                JensenShannon(display_suffix),
                TopKAccuracy(display_suffix, 2),
                TopKAccuracy(display_suffix, 3),
                TopKAccuracy(display_suffix, 5),
                DistanceMatrix(display_suffix),
                # ConfusionMatrix(display_suffix),
                Loss(display_suffix),
            ]
        else:
            self.metrics = [
                Accuracy(display_suffix),
                CategoricalAccuracy(display_suffix),
                JensenShannon(display_suffix),
                TopKAccuracy(display_suffix, 2),
                TopKAccuracy(display_suffix, 3),
                TopKAccuracy(display_suffix, 5),
                Loss(display_suffix),
            ]

    def __call__(self):
        results = {}
        for metric in self.metrics:
            results.update(metric())
        return results

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, predicted_logits: torch.Tensor, target_indices: torch.Tensor):
        assert predicted_logits.dim() == 3
        assert target_indices.dim() == 2
        for metric in self.metrics:
            metric.update(predicted_logits, target_indices)


###############################################################################
# Batch metrics
###############################################################################


class Accuracy:

    def __init__(self, display_suffix):
        self.display_suffix = display_suffix
        self.reset()

    def __call__(self):
        return {f'Accuracy/{self.display_suffix}': float((self.true_positives / self.count).cpu())}

    def reset(self):
        self.count = 0
        self.true_positives = 0

    def update(self, predicted_logits, target_indices):
        if ppgs.BACKEND is not None:
            predicted_logits = ppgs.BACKEND(predicted_logits)

        # Predicted category is the maximum logit
        predicted_indices = predicted_logits.argmax(dim=1)

        # Compare to target indices
        self.true_positives += torch.logical_and(predicted_indices == target_indices, target_indices != -100).sum()

        # Update count
        self.count += (target_indices != -100).sum()

class TopKAccuracy:

    def __init__(self, display_suffix: str, k: int):
        self.display_suffix = display_suffix
        self.k = k
        self.reset()

    def __call__(self):
        return {f'Top{self.k}Accuracy/{self.display_suffix}': float(float(self.correct_in_top_k) / float(self.count))}
    
    def reset(self):
        self.count = 0
        self.correct_in_top_k = 0

    def update(self, predicted_logits: torch.Tensor, target_indices: torch.Tensor):
        """Assumes that predicted_logits are BATCH x DIMS x TIME, target_indices are BATCH x TIME"""
        if ppgs.BACKEND is not None:
            predicted_logits = ppgs.BACKEND(predicted_logits)

        predicted_logits = predicted_logits.transpose(1, 2)
        predicted_logits = predicted_logits.flatten(0, 1)
        target_indices = target_indices.flatten()

        nonpad_indices = target_indices != -100
        predicted_logits = predicted_logits[nonpad_indices]
        target_indices = target_indices[nonpad_indices]

        top_k = torch.topk(predicted_logits, self.k, dim=-1)
        top_k_indices = top_k.indices
        self.correct_in_top_k += ((top_k_indices == target_indices[:, None])).sum()
        self.count += len(target_indices)


class CategoricalAccuracy:

    def __init__(self, display_suffix):
        self.display_suffix = display_suffix
        self.reset()
        self.map = {i: phoneme for i, phoneme in enumerate(ppgs.PHONEME_LIST)}

    def __call__(self):
        if self.totals is not None:
            assert self.totals.shape == self.counts.shape
        else:
            return None
        output = {}
        for i in range(0, self.totals.shape[0]):
            output[f"Accuracy/{self.display_suffix}/{self.map[i]}"] = (self.totals[i] / self.counts[i]).item()
            output[f"Total/{self.display_suffix}/{self.map[i]}"] = self.totals[i].item()
            output[f"Count/{self.display_suffix}/{self.map[i]}"] = self.counts[i].item()
        return output

    def reset(self):
        self.totals = None
        self.counts = None
    
    def update(self, predicted_logits, target_indices):
        """Update per-category accuracy"""

        if ppgs.BACKEND is not None:
            predicted_logits = ppgs.BACKEND(predicted_logits)

        #Unroll time dimensionality
        if len(predicted_logits.shape) == 3: #handle batched input
            predicted_logits = torch.transpose(predicted_logits, 1, 2) #Batch,Class,Time->Batch,Time,Class
            predicted_logits = torch.flatten(predicted_logits, 0, 1) #Batch,Time,Class->Batch*Time,Class
            target_indices = torch.flatten(target_indices) #Batch,Time->Batch*Time (1D)

        #deal with -100 ignore values
        keep_indices = (target_indices != -100)
        target_indices = target_indices[keep_indices]
        predicted_logits = predicted_logits[keep_indices]

        #convert logits to onehot
        predicted_indices = predicted_logits.argmax(dim=1)
        predicted_onehots = torch.nn.functional.one_hot(predicted_indices, num_classes=predicted_logits.shape[-1])

        #convert targets to onehot
        target_onehots = torch.nn.functional.one_hot(target_indices, num_classes=predicted_logits.shape[-1])

        #update (or set) totals
        marginal_totals = torch.mul(predicted_onehots, target_onehots).sum(dim=0)
        if self.totals is None:
            self.totals = marginal_totals
        else:
            self.totals += marginal_totals

        #update (or set) counts
        marginal_counts = target_onehots.sum(dim=0)
        if self.counts is None:
            self.counts = marginal_counts
        else:
            self.counts += marginal_counts


class Loss:

    def __init__(self, display_suffix, kind=ppgs.LOSS_FUNCTION):
        self.display_suffix = display_suffix
        self.kind = kind
        self.loss_fn = ppgs.train.Loss(kind=kind)
        self.reset()

    def __call__(self):
        if self.kind == 'CTC':
            return {f'Loss/{self.kind}/{self.display_suffix}': float((self.total).cpu().numpy())}
        else:
            return {f'Loss/{self.kind}/{self.display_suffix}': float((self.total / self.count).cpu().numpy())}

    def reset(self):
        self.total = 0.
        self.count = 0

    def update(self, predicted_logits, target_indices):
        """Update the total cross entropy loss"""
        self.total += self.loss_fn(
            predicted_logits,
            target_indices)

        # Update count
        self.count += (target_indices != -100).sum()

class JensenShannon:

    def __init__(self, display_suffix):
        self.display_suffix = display_suffix
        self.reset()

    def __call__(self):
        return {f'JSD/{self.display_suffix}': float((self.total / self.count).cpu().numpy())}

    def reset(self):
        self.total = 0.
        self.count = 0

    def update(self, predicted_logits, target_indices):
        """Update the total JSD"""

        if ppgs.BACKEND is not None:
            predicted_logits = ppgs.BACKEND(predicted_logits)
        
        #Unroll time dimensionality
        if len(predicted_logits.shape) == 3: #handle batched input
            predicted_logits = torch.transpose(predicted_logits, 1, 2) #Batch,Class,Time->Batch,Time,Class
            predicted_logits = torch.flatten(predicted_logits, 0, 1) #Batch,Time,Class->Batch*Time,Class
            target_indices = torch.flatten(target_indices) #Batch,Time->Batch*Time (1D)
        
        #deal with -100 ignore values
        keep_indices = (target_indices != -100)
        target_indices = target_indices[keep_indices]
        predicted_logits = predicted_logits[keep_indices]
        target_onehot = torch.nn.functional.one_hot(target_indices, num_classes=predicted_logits.shape[-1])

        #compute logits for targets
        target_logits = torch.special.logit(target_onehot, eps=1e-5)

        #calculate JSD and update totals
        self.total += jensenShannonDivergence(predicted_logits, target_logits, as_logits=True)
        self.count += (target_indices != -100).sum() #TODO investigate if -100 needs to be used in JSD THE ANSWER IS YES!!


class DistanceMatrix:

    def __init__(self, display_suffix, weighted=True):
        self.display_suffix = display_suffix
        if weighted:
            self.weights = torch.load(ppgs.CLASS_WEIGHT_FILE)
        else:
            self.weights = torch.ones(ppgs.OUTPUT_CHANNELS)
        self.reset()

    def _normalized(self):
        # col_normed = self.matrix / self.matrix.sum(dim=0)[None, :]
        # row_normed = torch.nn.functional.log_softmax(col_normed, dim=1)
        row_normed = self.matrix / self.matrix.sum(dim=1)[:, None]
        normalized = row_normed
        # total = self.count.sum()
        # scaler = self.count / total
        # for col, val in enumerate(scaler):
        #     normalized[:, col] /= val
        # import pdb; pdb.set_trace()
        # frequency_ratios = self.count / self.count.sum()
        # import pdb; pdb.set_trace()
        # normalized = probabilities / frequency_ratios
        return normalized.cpu()

    def _render(self):
        figure = plt.figure(dpi=400, figsize=(6, 6))
        ax = figure.add_subplot()
        mat = ax.matshow(self._normalized(), norm=PowerNorm(gamma=1/3))
        # mat = ax.matshow(self._normalized())
        phones = ppgs.PHONEME_LIST
        num_phones = len(ppgs.PHONEME_LIST)
        ax.locator_params('both', nbins=num_phones)
        ax.set_xticklabels([''] + phones, rotation='vertical')
        ax.set_yticklabels([''] + phones)
        ax.tick_params(axis='x', top=True, bottom=True, labelbottom=True, labeltop=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        ax.figure.colorbar(mat, cax=cax)
        figure.align_labels()

        phone_pairs = [
            ('f', 'v'),
            ('s', 'z'),
            ('sh', 'zh')
        ]

        padding = 0.5 + 0.2

        for phone0, phone1 in phone_pairs:
            idx0 = ppgs.PHONEME_LIST.index(phone0)
            idx1 = ppgs.PHONEME_LIST.index(phone1)
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
        return {
            f'DistanceMatrix/{self.display_suffix}': self._render(),
            f'DistanceMatrix/{self.display_suffix}/data': self._normalized()
        }

    def reset(self):
        self.matrix = None
        self.count = None

    def update(self, predicted_logits, target_indices):
        """Assumes that predicted_logits are BATCH x DIMS x TIME"""

        if ppgs.BACKEND is not None:
            predicted_logits = ppgs.BACKEND(predicted_logits)

        if self.weights.device != target_indices.device:
            self.weights = self.weights.to(target_indices.device)

        predicted_logits = predicted_logits.to(torch.float32)

        predicted_logits = torch.transpose(predicted_logits, 1, 2) #Batch,Class,Time->Batch,Time,Class
        predicted_logits = torch.flatten(predicted_logits, 0, 1) #Batch,Time,Class->Batch*Time,Class
        predicted_probs = torch.nn.functional.softmax(predicted_logits, dim=1)
        predicted_probs = torch.mul(predicted_probs, self.weights[None, :])
        predicted_indices = predicted_probs.argmax(dim=1)
        target_indices = torch.flatten(target_indices)

        nonpad_indices = target_indices != -100
        predicted_probs = predicted_probs[nonpad_indices]
        predicted_indices = predicted_indices[nonpad_indices]

        num_categories = predicted_probs.shape[-1]

        if self.matrix is None:
            self.matrix = torch.zeros((num_categories, num_categories)).to(device=predicted_logits.device, dtype=predicted_logits.dtype)

        if self.count is None:
            self.count = torch.zeros(num_categories).to(predicted_logits.device, dtype=torch.long)

        # for idx, probs in zip(predicted_indices, predicted_probs):
        #     self.matrix[idx] += probs
        self.matrix.scatter_add_(
            0,
            predicted_indices[:, None].expand(-1, predicted_probs.shape[-1]),
            predicted_probs)

        target_indices = target_indices[nonpad_indices]
        self.count += target_indices.bincount()

        # for probs, index in zip(predicted_probs, predicted_indices):
        #     self.matrix[index] += probs


class ConfusionMatrix:

    def __init__(self, display_suffix):
        self.display_suffix = display_suffix
        self.reset()

    def _normalized(self):
        return (self.matrix / self.matrix.sum(dim=1)).cpu()

    def _render(self):
        figure = plt.figure(dpi=400, figsize=(6, 6))
        ax = figure.add_subplot()
        ax.matshow(self._normalized())
        phones = ppgs.PHONEME_LIST
        num_phones = len(ppgs.PHONEME_LIST)
        ax.locator_params('both', nbins=num_phones)
        ax.set_xticklabels([''] + phones, rotation='vertical')
        ax.set_yticklabels([''] + phones)
        ax.set_ylabel('Ground Truth Phoneme')
        ax.set_xlabel('Model Predicted Probabilities')
        figure.align_labels()
        return figure

    def __call__(self):
        return {f'ConfusionMatrix/{self.display_suffix}': self._render()}
    
    def reset(self):
        self.matrix = None

    def update(self, predicted_logits, target_indices):
        """Assumes that predicted_logits are BATCH x DIMS x TIME"""

        if ppgs.BACKEND is not None:
            predicted_logits = ppgs.BACKEND(predicted_logits)

        predicted_logits = torch.transpose(predicted_logits, 1, 2) #Batch,Class,Time->Batch,Time,Class
        predicted_logits = torch.flatten(predicted_logits, 0, 1) #Batch,Time,Class->Batch*Time,Class
        predicted_probs = torch.nn.functional.softmax(predicted_logits, dim=1)
        target_indices = torch.flatten(target_indices)

        nonpad_indices = target_indices != -100
        predicted_probs = predicted_probs[nonpad_indices]
        target_indices = target_indices[nonpad_indices]

        num_categories = predicted_probs.shape[-1]

        if self.matrix is None:
            self.matrix = torch.zeros((num_categories, num_categories)).to(predicted_logits.device)

        self.matrix[target_indices] += predicted_probs

        # for probs, index in zip(predicted_probs, target_indices):
        #     self.matrix[index] += probs


###############################################################################
# Additional Metric Functions
###############################################################################
def jensenShannonDivergence(p_tensor, q_tensor, as_logits=False, reduction='sum'):
    """Computes the pointwise Jensen Shannon divergence between tensors sampled from P and Q
    Note that p_tensor and q_tensor are both (possibly batched) probability tensors, NOT in the log space
    unless as_logits=True, in which case BOTH p_tensor and q_tensor are taken as probability logits"""
    m_tensor = (p_tensor+q_tensor)/2
    if not as_logits:
        kl_pm = torch.nn.functional.kl_div(torch.log(m_tensor), p_tensor, reduction="none")
        kl_pm = torch.nan_to_num(kl_pm)
        if reduction == 'sum':
            kl_pm = kl_pm.sum(dim=-1)
        kl_qm = torch.nn.functional.kl_div(torch.log(m_tensor), q_tensor, reduction="none")
        kl_qm = torch.nan_to_num(kl_qm)
        if reduction == 'sum':
            kl_qm = kl_qm.sum(dim=-1)
    else:
        kl_pm = torch.nn.functional.kl_div(m_tensor, p_tensor, log_target=True, reduction="none")
        kl_pm = torch.nan_to_num(kl_pm)
        if reduction == 'sum':
            kl_pm = kl_pm.sum(dim=-1)
        kl_qm = torch.nn.functional.kl_div(m_tensor, q_tensor, log_target=True, reduction='none')
        kl_qm = torch.nan_to_num(kl_qm)
        if reduction == 'sum':
            kl_qm = kl_qm.sum(dim=-1)
    combined = (kl_pm+kl_qm)/2
    combined[combined < 0] = 0
    sqrt = torch.sqrt(combined)
    summed = sqrt.sum(dim=0)
    return summed


if __name__ == '__main__':
    #show that it is additive
    print(jensenShannonDivergence(torch.tensor([0.30, 0.50, 0.20]),
        torch.tensor([0.36, 0.48, 0.16])))
    print(jensenShannonDivergence(torch.tensor([0.85, 0.5, 0.1]),
        torch.tensor([1.0, 1e-6, 1e-6])))
    print(jensenShannonDivergence(
        torch.tensor([[0.85, 0.5, 0.1], [0.30, 0.50, 0.20]]),
        torch.tensor([[1.0, 1e-6, 1e-6], [0.36, 0.48, 0.16]]),
    ))

    #Show that it handles zero inputs
    print(jensenShannonDivergence(
        torch.tensor([0.0, 9.0, 1.9]),
        torch.tensor([0.0, -9.0, 10.0]),
    ))
    print(jensenShannonDivergence(
        torch.tensor([1e-9, 9.0, 1.9]),
        torch.tensor([1e-9, -9.0+1e-9, 10.0]),
    ))

    JSMetric = JensenShannon('test')
    input_logits = torch.special.logit(torch.tensor([[0.8, 0.15, 0.05]]))
    input_indices = torch.tensor([1])
    JSMetric.update(input_logits, input_indices)
    print(JSMetric.total)

    Top3 = TopKAccuracy('test', 3)
    Top2 = TopKAccuracy('test', 2)
    input_indices = torch.tensor([0, 4, 1, 3, 1, 2])
    input_logits = torch.cat([
        torch.special.logit(torch.tensor([[0.55, 0.16, 0.05, 0.1, 0.14]])),
        torch.special.logit(torch.tensor([[0.2, 0.25, 0.11, 0.35, 0.09]])),
        torch.special.logit(torch.tensor([[0.9, 0.01, 0.01, 0.01, 0.07]])),
        torch.special.logit(torch.tensor([[0.1, 0.3, 0.2, 0.15, 0.05]])),
        torch.special.logit(torch.tensor([[0.8, 0.1, 0.01, 0.01, 0.08]])),
        torch.special.logit(torch.tensor([[0.21, 0.19, 0.2, 0.21, 0.19]]))
    ])
    Top3.update(input_logits, input_indices)
    Top2.update(input_logits, input_indices)
    print(Top2.correct_in_top_k, Top2.count, Top2())
    print(Top3.correct_in_top_k, Top3.count, Top3())
