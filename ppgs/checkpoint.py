import torch


###############################################################################
# Checkpoint utilities
###############################################################################


def latest_path(directory, regex='*.pt'):
    """Retrieve the path to the most recent checkpoint"""
    # Retrieve checkpoint filenames
    files = list(directory.glob(regex))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Retrieve latest checkpoint
    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file.stem))))
    return files[-1]


def load(checkpoint_path, model, optimizer=None):
    """Load model checkpoint from file"""
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    # Restore model
    model.load_state_dict(checkpoint_dict['model'])

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    # Restore training state
    step = checkpoint_dict['step']
    epoch = checkpoint_dict['epoch']

    print("Loaded checkpoint '{}' (step {})" .format(checkpoint_path, step))

    return model, optimizer, step, epoch


def save(model, optimizer, step, epoch, checkpoint_path, accelerator = None):
    """Save training checkpoint to disk"""
    if accelerator is None:
        save_fn = torch.save
    else:
        save_fn = accelerator.save
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    save_fn(checkpoint, checkpoint_path)
