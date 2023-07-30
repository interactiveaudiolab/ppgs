import torch

import ppgs


def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        # ==========================================================================================================
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        # ==========================================================================================================
        probs = log_probs[:input_length, i].exp()
        # ==========================================================================================================
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
        # ==========================================================================================================
        losses.append(-alpha[-2:].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        return output.sum()
    output = output.to(dt)
    return output

def CTC(logits, targets):
    """Takes logits in shape BATCH x CATEGORIES x TIME and targets in shape BATCH x TIME"""
    # original_targets = targets.clone().cpu()
    # original_logits = logits.clone().cpu()
    logits = logits.to(torch.float32)
    # targets = targets.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1).permute(2, 0, 1)
    # trick to get sequence lengths
    lengths = torch.argmin(torch.nn.functional.pad(targets, (0, 1), value=-100), dim=1)
    # remove uniques
    # import pdb; pdb.set_trace()
    target_sequences = [torch.unique_consecutive(seq)[:-1] for seq in targets]
    target_lengths = torch.tensor([len(seq) for seq in target_sequences])
    padded_targets = torch.full((len(target_sequences), max(target_lengths)), -100, dtype=torch.long, device=logits.device)
    for idx, target_sequence in enumerate(target_sequences):
        padded_targets[idx, :target_lengths[idx]] = target_sequence
    loss = torch.nn.functional.ctc_loss(log_probs, padded_targets, lengths, target_lengths, blank=ppgs.OUTPUT_CHANNELS-1, zero_infinity=True, reduction='none')
    print(loss)
    # import pdb; pdb.set_trace()
    # loss = ctcloss_reference(log_probs, padded_targets, lengths, target_lengths, blank=len(ppgs.PHONEME_LIST)-1, reduction='none')
    mean_loss = loss.mean()
    print(mean_loss)
    # if loss.min() < 0:
    #     import pdb; pdb.set_trace()
    # try:
    #     print(loss)
    #     print(mean_loss)
    # except:
    #     import pdb; pdb.set_trace()
    # if loss.min() < 1:
    #     import pdb; pdb.set_trace()
    # if mean_loss < 1:
    #     import pdb; pdb.set_trace()
    return mean_loss


def Loss(kind=ppgs.LOSS_FUNCTION):
    if kind == 'CE':
        return torch.nn.functional.cross_entropy
    if kind == 'CTC':
        return CTC
    raise ValueError('Unknown loss function:', kind)
