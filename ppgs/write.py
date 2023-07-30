from matplotlib.figure import Figure
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter

import ppgs

###############################################################################
# Utilities
###############################################################################


# class PpgsSummaryWriter(SummaryWriter):

#     def add_visualization(self, tag, logits, global_step=None, close=True, walltime=None):


def writer(directory):
    """Get the writer object"""
    if not hasattr(writer, 'writer') or writer.directory != directory:
        writer.writer = SummaryWriter(log_dir=directory)
        writer.directory = directory
    return writer.writer

###############################################################################
# Tensorboard logging
###############################################################################


def audio(directory, step, audio):
    """Write audio to Tensorboard"""
    for name, waveform in audio.items():
        writer(directory).add_audio(
            name,
            waveform,
            step,
            ppgs.SAMPLE_RATE)

def metrics(directory, step, objects):
    """Write mixed objects to Tensorboard"""
    writer_object = writer(directory)
    for name, object in objects.items():
        if isinstance(object, Figure):
            writer_object.add_figure(name, object, step)
        else:
            writer_object.add_scalar(name, object, step)


def figures(directory, step, figures):
    """Write figures to Tensorboard"""
    for name, figure in figures.items():
        writer(directory).add_figure(name, figure, step)


def images(directory, step, images):
    """Write images to Tensorboard"""
    for name, image in images.items():
        writer(directory).add_image(name, image, step, dataformats='HWC')

def visualizations(directory, step, videos):
    """Write visualizations to Tensorboard"""
    for name, video in videos.items():
        writer(directory)._get_file_writer().add_summary(
            Summary(value=[Summary.Value(tag=name, image=video)]),
            global_step=step,
        )

def scalars(directory, step, scalars):
    """Write scalars to Tensorboard"""
    for name, scalar in scalars.items():
        writer(directory).add_scalar(name, scalar, step)
