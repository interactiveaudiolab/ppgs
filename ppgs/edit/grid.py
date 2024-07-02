import torch

import pypar

import ppgs


###############################################################################
# Grid-based interpolation
###############################################################################


def sample(ppg: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Grid-based PPG interpolation

    Arguments
        ppg
            Input PPG
        grid
            Grid of desired length; each item is a float-valued index into ppg

    Returns
        Interpolated PPG
    """
    # Get interpolation residuals
    interp = grid - torch.floor(grid)

    # Get PPG indices
    xp = torch.arange(ppg.shape[-1], device=ppg.device)
    i = torch.searchsorted(xp, grid, side='right')

    # Replicate final frame
    # "replication_pad1d_cpu" not implemented for 'Half'
    dtype = ppg.dtype
    if dtype in [torch.float16, torch.bfloat16]:
        ppg = torch.nn.functional.pad(
            ppg.to(torch.float32),
            (0, 1),
            mode='replicate'
        ).to(dtype)
    else:
        ppg = torch.nn.functional.pad(ppg, (0, 1), mode='replicate')

    # Linear interpolation
    return ppgs.interpolate(ppg[..., i - 1], ppg[..., i], interp)


###############################################################################
# Interpolation grids
###############################################################################


def constant(ppg: torch.Tensor, ratio: float) -> torch.Tensor:
    """Create a grid for constant-ratio time-stretching

    Arguments
        ppg
            Input PPG
        ratio
            Time-stretching ratio; lower is slower

    Returns
        Constant-ratio grid for time-stretching ppg
    """
    return of_length(ppg, round(ppg.shape[-1] / ratio + 1e-4))


def from_alignments(
    source: pypar.Alignment,
    target: pypar.Alignment,
    sample_rate: int = ppgs.SAMPLE_RATE,
    hopsize: int = ppgs.HOPSIZE
) -> torch.Tensor:
    """Create time-stretch grid to convert source alignment to target

    Arguments
        source
            Forced alignment of PPG to stretch
        target
            Forced alignment of target PPG
        sample_rate
            Audio sampling rate
        hopsize
            Hopsize in samples

    Returns
        Grid for time-stretching source PPG
    """
    # Get number of source and target frames
    source_frames = int((source.duration() * sample_rate) / hopsize)
    target_frames = int((target.duration() * sample_rate) / hopsize)

    # Get relative rate at each frame
    rates = pypar.compare.per_frame_rate(
        target,
        source,
        sample_rate,
        hopsize,
        target_frames)

    # Convert rates to indices and align edges
    indices = torch.cumsum(torch.tensor(rates), 0)
    indices -= indices[0].clone()
    indices *= (source_frames - 1) / indices[-1]

    return indices


def of_length(ppg: torch.Tensor, length: int) -> torch.Tensor:
    """Create time-stretch grid to resample PPG to a specified length

    Arguments
        ppg
            Input PPG
        length
            Target length

    Returns
        Grid of specified length for time-stretching ppg
    """
    return torch.linspace(
        0.,
        ppg.shape[-1] - 1.,
        length,
        dtype=torch.float,
        device=ppg.device)
