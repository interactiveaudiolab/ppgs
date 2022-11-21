import torch
import ppgs

###############################################################################
# Aggregate batch metric state
###############################################################################


class Metrics:

    def __init__(self, display_prefix):
        self.metrics = [
            Accuracy(display_prefix),
            CategoricalAccuracy(display_prefix),
            Loss(display_prefix),
            JensenShannon(display_prefix)
        ]

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
        return {f'{self.display_prefix}_accuracy': float((self.true_positives / self.count).cpu())}

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

class CategoricalAccuracy:
    
    def __init__(self, display_prefix):
        self.display_prefix = display_prefix
        self.reset()
        self.map = {i: phoneme for i, phoneme in enumerate(ppgs.PHONEME_LIST)}

    def __call__(self):
        if self.totals is not None:
            assert self.totals.shape == self.counts.shape
        else:
            return None
        output = {}
        for i in range(0, self.totals.shape[0]):
            output[f"{self.display_prefix}_phoneme_{self.map[i]}_accuracy"] = (self.totals[i] / self.counts[i]).item()
            output[f"{self.display_prefix}_phoneme_{self.map[i]}_total"] = self.totals[i].item()
            output[f"{self.display_prefix}_phoneme_{self.map[i]}_count"] = self.counts[i].item()
        return output

    def reset(self):
        self.totals = None
        self.counts = None
    
    def update(self, predicted_logits, target_indices):
        """Update per-category accuracy"""

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

    def __init__(self, display_prefix):
        self.display_prefix = display_prefix
        self.reset()

    def __call__(self):
        return {f'{self.display_prefix}_loss': float((self.total / self.count).cpu().numpy())}

    def reset(self):
        self.total = 0.
        self.count = 0

    def update(self, predicted_logits, target_indices):
        """Update the total cross entropy loss"""
        self.total += torch.nn.functional.cross_entropy(
            predicted_logits,
            target_indices,
            reduction='sum')

        # Update count
        self.count += (target_indices != -100).sum()

class JensenShannon:

    def __init__(self, display_prefix):
        self.display_prefix = display_prefix
        self.reset()

    def __call__(self):
        return {f'{self.display_prefix}_JSD': float((self.total / self.count).cpu().numpy())}

    def reset(self):
        self.total = 0.
        self.count = 0

    def update(self, predicted_logits, target_indices):
        """Update the total JSD"""
        
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

###############################################################################
# Additional Metric Functions
###############################################################################
def jensenShannonDivergence(p_tensor, q_tensor, as_logits=False):
    """Computes the pointwise Jensen Shannon divergence between tensors sampled from P and Q
    Note that p_tensor and q_tensor are both (possibly batched) probability tensors, NOT in the log space
    unless as_logits=True, in which case BOTH p_tensor and q_tensor are taken as probability logits"""
    m_tensor = (p_tensor+q_tensor)/2
    if not as_logits:
        kl_pm = torch.nn.functional.kl_div(torch.log(m_tensor), p_tensor, reduction="none")
        kl_pm = torch.nan_to_num(kl_pm).sum(dim=-1)
        kl_qm = torch.nn.functional.kl_div(torch.log(m_tensor), q_tensor, reduction="none")
        kl_qm = torch.nan_to_num(kl_qm).sum(dim=-1)
    else:
        kl_pm = torch.nn.functional.kl_div(m_tensor, p_tensor, log_target=True, reduction="none")
        kl_pm = torch.nan_to_num(kl_pm).sum(dim=-1)
        kl_qm = torch.nn.functional.kl_div(m_tensor, q_tensor, log_target=True, reduction='none')
        kl_qm = torch.nan_to_num(kl_qm).sum(dim=-1)
    return torch.sqrt((kl_pm+kl_qm)/2).sum(dim=0)


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
