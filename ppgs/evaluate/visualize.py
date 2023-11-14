import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pypar
import torch
import torchaudio
import torchutil
from moviepy import editor as mpy
from PIL import Image, ImageDraw, ImageFont

import ppgs


###############################################################################
# Constants
###############################################################################


# Display size
DISPLAY_WINDOW_SIZE = ppgs.SAMPLE_RATE // ppgs.HOPSIZE
DISPLAY_HOPSIZE = 2
DISPLAY_PADDING = DISPLAY_WINDOW_SIZE // 2 - DISPLAY_HOPSIZE//2


###############################################################################
# Visualize PPGs from audio input
###############################################################################


def from_audio_file_to_file(
    audio_filename,
    output_filename,
    textgrid_filename=None,
    checkpoint=None,
    font_filename=None,
    prepocess_only=False,
    gpu=None,
    mode='video'):
    """Create PPG visuals from an audio file and save"""
    # Infer PPG
    ppg = ppgs.from_file(
        audio_filename,
        checkpoint=checkpoint,
        gpu=gpu
    ).squeeze(dim=0).T.cpu()

    # Visualize
    if mode == 'video':
        from_ppg_to_video_file(
            ppg,
            audio_filename,
            output_filename,
            textgrid_filename=textgrid_filename)
    elif mode == 'image':
        from_ppg_to_image_file(
            ppg,
            audio_filename,
            output_filename,
            textgrid_filename=textgrid_filename,
            font_filename=font_filename)
    raise ValueError(mode)


def from_audio_files_to_files(
    audio_filenames,
    output_dir,
    textgrid_filename=None,
    checkpoint=None,
    font_filename=None,
    preprocess_only=False,
    gpu=None,
    mode='video'):
    """Create PPG visuals from audio files and save"""
    for audio_filename in torchutil.iterator(
        audio_filenames,
        'Creating visualizations',
        total=len(audio_filenames)
    ):
        if mode == 'video':
            from_audio_file_to_file(
                audio_filename,
                str(Path(output_dir) / (Path(audio_filename).stem + '.mp4')),
                textgrid_filename=textgrid_filename,
                checkpoint=checkpoint,
                prepocess_only=preprocess_only,
                gpu=gpu)
        elif mode == 'image':
            from_audio_file_to_file(
                audio_filename,
                str(Path(output_dir) / (Path(audio_filename).stem + '.jpg')),
                textgrid_filename=textgrid_filename,
                checkpoint=checkpoint,
                font_filename=font_filename,
                prepocess_only=preprocess_only,
                gpu=gpu,
                mode='image')


###############################################################################
# Visualize PPGs from PPG input
###############################################################################


def from_ppg_to_image_file(
    ppg,
    audio_filename,
    image_filename,
    textgrid_filename=None,
    second_ppg=None,
    font_filename=None,
    labels=ppgs.PHONEMES,
    scalefactor=(32, 32),
    padding=None):
    """Visualize PPG"""
    # Load audio
    audio = torchaudio.load(audio_filename)[0][0]

    # Visualize PPG
    if padding is None:
        padding = 5 * scalefactor[1] // scalefactor[0]
    ppg_pixels = from_ppg_to_pixels(ppg, padding=padding)

    # Visualize one-hot
    if textgrid_filename is None:
        alignment_pixels = None
    else:
        alignment_pixels = from_textgrid_to_pixels(
            textgrid_filename,
            len(audio) // ppgs.HOPSIZE,
            padding=padding)

    # Maybe overlay another PPG
    if second_ppg is None:
        combined = combine_pixels(ppg_pixels, alignment_pixels)
    else:
        ppg2_pixels = from_ppg_to_pixels(second_ppg, padding=padding)
        combined = combine_pixels(
            ppg_pixels,
            green=alignment_pixels,
            blue=ppg2_pixels)
    combined = combined.permute(1, 0, 2).to(torch.uint8)
    combined = combined.numpy()

    # Resize image
    image = Image.fromarray(combined)
    image = image.resize(
        (image.width * scalefactor[0], image.height * scalefactor[1]),
        Image.NEAREST)

    # Add labels
    if font_filename is not None:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_filename, size=scalefactor[1])
        for idx, label in enumerate(labels):
            draw.text(
                (scalefactor[0] // 2, idx * scalefactor[1]),
                label,
                font=font)

    # Save
    image.save(image_filename)


def from_ppg_to_video_file(
    ppg,
    audio_filename,
    video_filename,
    textgrid_filename=None,
    preprocess_only=False,
    labels=ppgs.PHONEMES,
    scalefactor=16):
    """Takes ppg of shape time,categories and creates a visualization"""
    # Load audio
    audio = torchaudio.load(audio_filename)[0][0]
    audio_clip = mpy.AudioFileClip(audio_filename, fps=ppgs.SAMPLE_RATE)

    num_frames = len(audio) // ppgs.HOPSIZE

    ppg_pixels = from_ppg_to_pixels(ppg)
    textgrid_pixels = from_textgrid_to_pixels(
        textgrid_filename,
        num_frames,
    )

    pixels = combine_pixels(ppg_pixels, textgrid_pixels)

    # Chunk PPG into video frames
    frames = []
    for i in range(0, (audio.shape[0] // ppgs.HOPSIZE) // DISPLAY_HOPSIZE):

        # Chunk
        left = i * DISPLAY_HOPSIZE
        right = left + DISPLAY_WINDOW_SIZE
        frame = pixels[left:right, :].transpose(0, 1)

        # Add black bar at bottom
        frame = torch.cat(
            [frame, torch.zeros((10, frame.shape[1], frame.shape[2]))])

        # Add frame
        frames.append(frame.numpy())

    # Create clip
    clip = mpy.ImageSequenceClip(
        frames,
        fps=DISPLAY_WINDOW_SIZE // DISPLAY_HOPSIZE)

    # Resize
    clip = clip.fl_image(lambda frame: resizer(frame, scalefactor))

    # Cache label + playhead overlay to improve performance
    if not hasattr(from_ppg_to_video_file, 'overlay'):

        text_clips = []
        if not preprocess_only:
            for i, label in enumerate(labels):

                # Create text
                text_clip = mpy.TextClip(
                    label,
                    color='rgb(255,255,255)',
                    fontsize=scalefactor,
                    bg_color='black')

                # Brighten text
                text_clip = text_clip.fl_image(
                    lambda frame: brighten(frame, 1.5))

                # Set text location
                text_clip = text_clip.set_position((
                    clip.size[0] // 2 - text_clip.size[0] - scalefactor,
                    scalefactor * i + 1))

                # Add text
                text_clips.append(text_clip)

        # Create playhead
        playhead = np.zeros((clip.size[1], 1, 3))
        playhead[:, 0, 0] = np.full(clip.size[1], 255)
        overlay_clip = mpy.ImageClip(playhead)

        # Set playhead to persist throughout clip
        overlay_clip = overlay_clip.set_duration(clip.duration)

        # Set playhead position
        overlay_clip = overlay_clip.set_position(
            (clip.size[0] // 2 - scalefactor, 0))

        # Create overlay
        blank = mpy.ColorClip(
            clip.size,
            color=(0.0, 0.0, 0.0),
            duration=clip.duration
        ).set_fps(1).set_opacity(0)

        # Overlay text and playhead clips (slow)
        overlay = mpy.CompositeVideoClip(
            [blank, overlay_clip] + text_clips)

        # Set overlay to persist throughout clip
        overlay = overlay.set_duration(clip.duration)

        # Render (slow)
        overlay = mpy.ImageSequenceClip(list(overlay.iter_frames()), fps=1)

        # Create mask over unused pixels
        overlay_mask = overlay.copy().to_mask()
        overlay_mask = mpy.ImageClip(
            np.where(overlay_mask.get_frame(0) > 0, 1.0, 0.0),
            ismask=True
        ).set_duration(clip.duration)

        # Apply mask
        overlay = overlay.set_mask(overlay_mask)

        # Cache
        from_ppg_to_video_file.overlay = overlay

    # Set overlay to persist throughout clip
    from_ppg_to_video_file.overlay.set_duration(clip.duration)

    # Apply overlay to clip
    composite = mpy.CompositeVideoClip([clip, from_ppg_to_video_file.overlay])

    # Clean-up non-overlaid clip
    clip.close()

    # Write audio
    composite = composite.set_audio(audio_clip)

    # Save to disk
    composite.write_videofile(
        video_filename,
        preset='ultrafast',
        audio_codec='aac',
        threads=8)
    composite.close()


def from_ppg_file_to_file(
    ppg_filename,
    audio_filename,
    output_filename,
    textgrid_filename=None,
    second_ppg_filename=None,
    font_filename=None,
    mode='image'):
    """Visualize PPG file and save"""
    # Load PPG
    ppg = torch.load(ppg_filename).T

    # Maybe overlay another PPG
    if second_ppg_filename is not None:
        ppg2 = torch.load(second_ppg_filename).T
    else:
        ppg2 = None

    # Visualize
    if mode == 'image':
        from_ppg_to_image_file(
            ppg=ppg,
            audio_filename=audio_filename,
            image_filename=output_filename,
            textgrid_filename=textgrid_filename,
            second_ppg=ppg2,
            font_filename=font_filename)


def from_ppg_to_video(ppg, audio_filename, labels=ppgs.PHONEMES):
    """Animate PPG"""
    # Create temporary video file
    temp_filename = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False).name

    # Create visual and save to disk
    from_ppg_to_video_file(
        ppg=ppg,
        audio_filename=audio_filename,
        video_filename=temp_filename,
        labels=labels)

    # Load from disk
    with open(temp_filename, 'rb') as video_file:
        video = video_file.read()

    # Cleanup temporary file
    os.remove(temp_filename)

    return video


def from_ppgs_to_videos(ppgs, audio_filenames, labels=ppgs.PHONEMES):
    """Animate PPGs"""
    return [
        from_ppg_to_video(ppg, file, labels=labels)
        for ppg, file in zip(ppgs, audio_filenames)]


###############################################################################
# Utilities
###############################################################################


def brighten(pic, factor):
    """Brighten image"""
    return np.clip(pic * factor, 0.0, 255.0)


def combine_pixels(red, blue=None, green=None):
    """Maybe merge PPG pixels by splitting color channels"""
    combined = torch.clone(red)

    #clear all but red channels
    combined[..., 1:] = 0
    if blue is not None:
        combined[..., 2] = blue[..., 2]
    if green is not None:
        combined[..., 1] = green[..., 1]

    return combined


def from_textgrid_to_pixels(
    textgrid_filename,
    num_frames,
    num_phonemes=len(ppgs.PHONEMES),
    padding=DISPLAY_PADDING):
    """Convert textgrid alignment to one-hot image data"""
    # Load alignment
    alignment = pypar.Alignment(textgrid_filename)

    # Get time corresponding to each frame
    hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
    times = np.linspace(
        hopsize / 2,
        (num_frames - 1) * hopsize + hopsize / 2,
        num_frames)
    times[-1] = alignment.duration()

    # Upsample phonemes to frame resolution
    phonemes = torch.tensor(
        alignment.framewise_phoneme_indices(
            ppgs.PHONEME_TO_INDEX_MAPPING,
            hopsize,
            times),
        dtype=torch.long)

    # Convert to one-hot
    phonemes = torch.nn.functional.one_hot(
        phonemes,
        num_classes=num_phonemes)

    # Convert one-hot to image data
    pixels = phonemes * 255 # scale to [0, 255]
    pixels = torch.nn.functional.pad(pixels, (0, 0, padding, padding))
    return pixels.unsqueeze(-1).repeat(1, 1, 3)


def from_ppg_to_pixels(ppg, padding=DISPLAY_PADDING):
    """Convert PPG to image data"""
    ppg = ppg.to(torch.float)
    if ppg.dim() == 3:
        ppg = ppg.squeeze(dim=0).T

    # Scale to image range
    pixels = ppg * 255

    # Pad so playhead is centered
    pixels = torch.nn.functional.pad(pixels, (0, 0, padding, padding))

    # Unsqueeze and convert form greyscale to rgb
    return pixels.unsqueeze(-1).repeat(1,1,3)


def resizer(pic, factor):
    """Nearest neighbors image resizing"""
    return cv2.resize(
        pic.astype('uint8'),
        (pic.shape[1] * factor, pic.shape[0] * factor),
        interpolation=cv2.INTER_NEAREST)
