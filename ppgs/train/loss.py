import ppgs
import torch

def CTC(logits, targets):
    """Takes logits in shape BATCH x CATEGORIES x TIME and targets in shape BATCH x TIME"""
    original_targets = targets.clone().cpu()
    original_logits = logits.clone().cpu()
    logits = logits.to(torch.float32)
    targets = targets.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1).permute(2, 0, 1)
    # trick to get sequence lengths
    lengths = torch.argmin(torch.nn.functional.pad(targets, (0, 1), value=-100), dim=1)
    # remove uniques
    target_sequences = [torch.unique_consecutive(seq) for seq in targets]
    target_lengths = torch.tensor([len(seq) for seq in target_sequences])
    padded_targets = torch.full((len(target_sequences), max(target_lengths)), -100, dtype=torch.long, device=logits.device)
    for idx, target_sequence in enumerate(target_sequences):
        padded_targets[idx, :target_lengths[idx]] = target_sequence
    loss = torch.nn.functional.ctc_loss(log_probs, padded_targets, lengths, target_lengths, blank=len(ppgs.PHONEME_LIST)-1, zero_infinity=True, reduction='none')
    mean_loss = loss.mean()
    try:
        print(loss)
        print(mean_loss)
    except:
        import pdb; pdb.set_trace()
    if loss.min() < 1:
        import pdb; pdb.set_trace()
    if mean_loss < 1:
        import pdb; pdb.set_trace()
    return mean_loss


def Loss(kind=ppgs.LOSS_FUNCTION):
    if kind == 'CE':
        return torch.nn.functional.cross_entropy
    if kind == 'CTC':
        return CTC
    raise ValueError('Unknown loss function:', kind)
