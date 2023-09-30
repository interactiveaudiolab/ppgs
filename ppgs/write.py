import accelerate
from matplotlib.figure import Figure

import ppgs


###############################################################################
# Utilities
###############################################################################


STATE = accelerate.state.PartialState()

@STATE.on_main_process
def writer(directory):
    """Get the writer object"""
    if not hasattr(writer, 'writer') or writer.directory != directory:
        from torch.utils.tensorboard import SummaryWriter
        writer.writer = SummaryWriter(log_dir=directory)
        writer.directory = directory
    return writer.writer


###############################################################################
# Tensorboard logging
###############################################################################


@STATE.on_main_process
def audio(directory, step, audio):
    """Write audio to Tensorboard"""
    for name, waveform in audio.items():
        writer(directory).add_audio(
            name,
            waveform,
            step,
            ppgs.SAMPLE_RATE)


@STATE.on_main_process
def metrics(directory, step, objects):
    """Write mixed objects to Tensorboard"""
    writer_object = writer(directory)
    for name, object in objects.items():
        if isinstance(object, Figure):
            writer_object.add_figure(name, object, step)
        else:
            writer_object.add_scalar(name, object, step)


@STATE.on_main_process
def figures(directory, step, figures):
    """Write figures to Tensorboard"""
    for name, figure in figures.items():
        writer(directory).add_figure(name, figure, step)


@STATE.on_main_process
def images(directory, step, images):
    """Write images to Tensorboard"""
    for name, image in images.items():
        writer(directory).add_image(name, image, step, dataformats='HWC')


@STATE.on_main_process
def visualizations(directory, step, videos):
    """Write visualizations to Tensorboard"""
    from tensorboard.compat.proto.summary_pb2 import Summary
    for name, video in videos.items():
        writer(directory)._get_file_writer().add_summary(
            Summary(value=[Summary.Value(tag=name, image=video)]),
            global_step=step)


@STATE.on_main_process
def scalars(directory, step, scalars):
    """Write scalars to Tensorboard"""
    for name, scalar in scalars.items():
        writer(directory).add_scalar(name, scalar, step)
