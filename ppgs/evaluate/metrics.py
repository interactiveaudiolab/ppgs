import torch


###############################################################################
# Aggregate batch metric state
###############################################################################


class Metrics:

    def __init__(self, display_prefix):
        self.metrics = [Accuracy(display_prefix), Loss(display_prefix)]

    def __call__(self):
        results = {}
        for metric in self.metrics:
            results.update(metric())
        return results

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, predicted_logits, target_indices):
        for metric in self.metrics:
            metric.update(predicted_logits, target_indices)


###############################################################################
# Batch metrics
###############################################################################


class Accuracy:

    def __init__(self, display_prefix):
        self.display_prefix = display_prefix
        self.reset()

    def __call__(self):
        return {f'{self.display_prefix}_accuracy': self.true_positives / self.count}

    def reset(self):
        self.count = 0
        self.true_positives = 0

    def update(self, predicted_logits, target_indices):
        # Predicted category is the maximum logit
        predicted_indices = predicted_logits.argmax(dim=1)

        # Compare to target indices
        self.true_positives += (predicted_indices == target_indices).sum()

        # Update count
        self.count += (target_indices != -100).sum()


class Loss:

    def __init__(self, display_prefix):
        self.display_prefix = display_prefix
        self.reset()

    def __call__(self):
        return {f'{self.display_prefix}_loss': self.total / self.count}

    def reset(self):
        self.total = 0.
        self.count = 0

    def update(self, predicted_logits, target_indices):
        # Update the total cross entropy loss
        self.total += torch.nn.functional.cross_entropy(
            predicted_logits,
            target_indices,
            reduction='sum')

        # Update count
        self.count += (target_indices != -100).sum()
