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
    xp = torch.arange(ppg.shape[-1], device=ppg.device)
    i = torch.clip(torch.searchsorted(xp, grid, right=True), 1, len(xp) - 1)

    # Linear interpolation
    # return (
    #     (ppg[..., i - 1] * (xp[i] - grid) + ppg[..., i] * (grid - xp[i - 1])) /
    #     (xp[i] - xp[i - 1]))

    # Spherical linear interpolation
    # TODO - is this correct?
    # TODO - should we center this (e.g., using i - 1, i, and i + 1)?
    return ppgs.interpolate(
        xp[i] - xp[i - 1],
        ppg[..., i - 1] * (xp[i] - grid),
        ppg[..., i] * (grid - xp[i - 1]))


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
    return torch.linspace(
        0.,
        ppg.shape[-1] - 1,
        round((ppg.shape[-1]) / ratio + 1e-4),
        dtype=torch.float,
        device=ppg.device)


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
    source_frames = (source.duration() * sample_rate) // hopsize
    target_frames = (target.duration() * sample_rate) // hopsize

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
