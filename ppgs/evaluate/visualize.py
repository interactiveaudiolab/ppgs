import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyfoal
import pypar
import torch
import torchaudio
import tqdm
from moviepy import editor as mpy
from PIL import Image, ImageDraw, ImageFont

import ppgs
from ppgs.data.dataset import ppgs_phoneme_list

#TODO make config safe (get values dynamically)

#Function to compute resize using cv2 Nearest Neighbor (preserve squares in distribution)
def resizer(pic, factor):
    new_size = (pic.shape[1]*factor, pic.shape[0]*factor)
    return cv2.resize(+pic.astype('uint8'), new_size, interpolation=cv2.INTER_NEAREST)

def brighten(pic, factor):
    return np.clip(pic * factor, 0.0, 255.0)

def emphasis_brighten(pic, exp):
    return np.clip(pic ** exp, 0.0, 255.0)

def invert(pic):
    return -1 * pic + 255.0

def logScale(pic):
    return np.clip(np.log(pic) * 255/np.log(255), 0.0, 255.0)

def expScale(pic):
    return np.clip((np.exp(pic*np.log(2))-1)*255.0, 0.0, 255.0)

def rootScale(pic):
    return np.clip(((pic/255.0) ** 0.625) * 255.0, 0.0, 255.0)

#Tunable parameters for appearance of generated video
display_window_size = ppgs.SAMPLE_RATE//ppgs.HOPSIZE
display_hopsize = 2
pad = display_window_size//2 - display_hopsize//2
scalefactor = 16
text_vertical_offset = 1

def from_textgrid_to_pixels(textgrid_filename, num_frames, num_phonemes=len(ppgs.PHONEME_LIST), padding=pad):
    alignment = pypar.Alignment(textgrid_filename)
    hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
    times = np.linspace(hopsize/2, (num_frames-1)*hopsize+hopsize/2, num_frames)
    times[-1] = alignment.duration()
    with ppgs_phoneme_list():
        phonemes = torch.tensor(pyfoal.convert.alignment_to_indices(
            alignment,
            hopsize=hopsize,
            return_word_breaks=False,
            times=times
        ), dtype=torch.long)
        phonemes = torch.nn.functional.one_hot(phonemes, num_classes=num_phonemes)

    pixels = phonemes * 255 #scale to [0,255]
    pixels = torch.nn.functional.pad(pixels, (0, 0, padding, padding))
    pixels = pixels.unsqueeze(-1).repeat(1, 1, 3)

    return pixels

def from_ppg_to_pixels(ppg, padding=pad):
    ppg = ppg.to(torch.float)
    pixels = ppg * 255 #scale to [0,255]
    # pixels = torch.nn.functional.one_hot(torch.argmax(ppg, dim=1), ppg.shape[-1]) * 255
    pixels = torch.nn.functional.pad(pixels, (0, 0, padding, padding)) #pad so playhead is centered
    pixels = pixels.unsqueeze(-1).repeat(1,1,3) #unsqueeze and convert form greyscale to rgb
    return pixels

def combine_pixels(red, blue=None, green=None):
    combined = torch.clone(red)

    #clear all but red channels
    combined[..., 1:] = 0
    if blue is not None:
        combined[..., 2] = blue[..., 2]
    if green is not None:
        combined[..., 1] = green[..., 1]

    return combined

def from_ppg_to_image_file(ppg, audio_filename, image_filename, textgrid_filename=None, second_ppg=None, font_filename=None, labels=ppgs.PHONEME_LIST, scalefactor=(32, 32), padding=None):
    audio = torchaudio.load(audio_filename)[0][0]
    num_frames = len(audio) // ppgs.HOPSIZE
    if padding is None:
        padding = 5*scalefactor[1]//scalefactor[0]
    ppg_pixels = from_ppg_to_pixels(ppg, padding=padding)
    alignment_pixels = from_textgrid_to_pixels(textgrid_filename, num_frames, padding=padding) if textgrid_filename is not None else None
    if second_ppg is None:
        combined = combine_pixels(ppg_pixels, alignment_pixels)
    else:
        ppg2_pixels = from_ppg_to_pixels(second_ppg, padding=padding)
        combined = combine_pixels(ppg_pixels, green=alignment_pixels, blue=ppg2_pixels)
    combined = combined.permute(1, 0, 2).to(torch.uint8)
    combined = combined.numpy()
    image = Image.fromarray(combined)
    image = image.resize((image.width*scalefactor[0], image.height*scalefactor[1]), Image.NEAREST)
    
    #add labels
    if font_filename is not None:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_filename, size=scalefactor[1])
        for idx, label in enumerate(labels):
            draw.text((scalefactor[0]//2, idx*scalefactor[1]), label, font=font)

    image.save(image_filename)


#TODO make scalefactor a parameter (currently is a hardcoded constant)
def from_ppg_to_video_file(ppg, audio_filename, video_filename, textgrid_filename=None, preprocess_only=False, labels=ppgs.PHONEME_LIST):
    """Takes ppg of shape time,categories and creates a visualization"""

    audio = torchaudio.load(audio_filename)[0][0]
    audio_clip = mpy.AudioFileClip(audio_filename, fps=ppgs.SAMPLE_RATE)

    num_frames = len(audio) // ppgs.HOPSIZE

    pixels = from_ppg_to_pixels(ppg, num_frames)

    #visual 'convolution' to create frames from ppg windows
    frames = []
    for i in range(0, (audio.shape[0]//ppgs.HOPSIZE)//display_hopsize):
        left = i * display_hopsize
        right = left+display_window_size
        frame = pixels[left:right,:].transpose(0, 1) #get a window of distributions
        frame = torch.cat([frame, torch.zeros((10, frame.shape[1], frame.shape[2]))]) #add black bar at bottom
        frames.append(frame.numpy())

    clip = mpy.ImageSequenceClip(frames, fps=display_window_size//display_hopsize) #create clip
    # clip = clip.fl_image(rootScale)
    # clip = clip.fl_image(invert)
    clip = clip.fl_image(lambda frame: resizer(frame, scalefactor)) #apply scaler filter
    
    if not hasattr(from_ppg_to_video_file, 'overlay'):
        #Create overlay on first call, then cache

        #Create labels only once
        text_clips = []
        if not preprocess_only:
            text_vertical_offset = 1
            for i, label in enumerate(labels):
                #create label text clip
                text_clip = mpy.TextClip(label, color="rgb(255,255,255)", fontsize=scalefactor, bg_color='black')
                text_clip = text_clip.fl_image(lambda frame: brighten(frame, 1.5))
                text_clip = text_clip.set_position((clip.size[0]//2-text_clip.size[0]-scalefactor,scalefactor*i+text_vertical_offset)) #position label vertically
                # text_clip = text_clip.set_duration(clip.duration) #give same duration
                text_clips.append(text_clip)

        #Create playhead
        playhead = np.zeros((clip.size[1], 1, 3))
        playhead[:,0,0] = np.full(clip.size[1], 255)
        overlay_clip = mpy.ImageClip(playhead)
        overlay_clip = overlay_clip.set_duration(clip.duration)
        overlay_clip = overlay_clip.set_position((clip.size[0]//2-scalefactor, 0))

        #Finally, create overlay
        blank = mpy.ColorClip(clip.size, color=(0.0, 0.0, 0.0), duration=clip.duration).set_fps(1).set_opacity(0)
        overlay = mpy.CompositeVideoClip([blank, overlay_clip]+text_clips) #composite (expensive)
        overlay = overlay.set_duration(clip.duration)
        overlay = mpy.ImageSequenceClip(list(overlay.iter_frames()), fps=1) #'render' (also expensive)
        overlay_mask = overlay.copy().to_mask() #create mask
        overlay_mask = mpy.ImageClip(np.where(overlay_mask.get_frame(0)>0, 1.0, 0.0), ismask=True).set_duration(clip.duration) #make mask all-or-nothing
        overlay = overlay.set_mask(overlay_mask) #apply mask

        from_ppg_to_video_file.overlay = overlay

    from_ppg_to_video_file.overlay.set_duration(clip.duration)
    composite = mpy.CompositeVideoClip([clip, from_ppg_to_video_file.overlay])
    clip.close()
    composite = composite.set_audio(audio_clip)

    #TODO add audio_codec as a parameter
    # composite.write_videofile(video_filename, audio_codec='aac', threads=8, logger=None)
    composite.write_videofile(video_filename, preset='ultrafast', audio_codec='aac', threads=8)
    composite.close()

def from_ppg_file_to_file(ppg_filename, audio_filename, output_filename, textgrid_filename=None, second_ppg_filename=None, font_filename=None, mode='image'):
    ppg = torch.load(ppg_filename).T
    if second_ppg_filename is not None:
        ppg2 = torch.load(second_ppg_filename).T
    else:
        ppg2 = None
    if mode == 'image':
        from_ppg_to_image_file(
            ppg=ppg,
            audio_filename=audio_filename,
            image_filename=output_filename,
            textgrid_filename=textgrid_filename,
            second_ppg=ppg2,
            font_filename=font_filename
        )

def from_audio_file_to_file(audio_filename, output_filename, textgrid_filename=None, checkpoint=ppgs.DEFAULT_CHECKPOINT, font_filename=None, prepocess_only=False, gpu=None, mode='video'):
    ppg = ppgs.from_file(audio_filename, checkpoint=checkpoint, gpu=gpu).squeeze(dim=0).T
    if mode == 'video':
        from_ppg_to_video_file(ppg.cpu(), audio_filename, output_filename, textgrid_filename=textgrid_filename)
    elif mode == 'image':
        from_ppg_to_image_file(ppg.cpu(), audio_filename, output_filename, textgrid_filename=textgrid_filename, font_filename=font_filename)
    else:
        raise ValueError(mode)


def from_audio_files_to_files(audio_filenames, output_dir, textgrid_filename=None, checkpoint=ppgs.DEFAULT_CHECKPOINT, font_filename=None, preprocess_only=False, gpu=None, mode='video'):
    iterator = tqdm.tqdm(
        audio_filenames,
        desc='Creating visualizations',
        total=len(audio_filenames),
        dynamic_ncols=True
    )
    for audio_filename in iterator:
        if mode == 'video':
            output_filename = str(Path(output_dir) / (Path(audio_filename).stem + '.mp4'))
            from_audio_file_to_file(audio_filename, output_filename, textgrid_filename=textgrid_filename, checkpoint=checkpoint, prepocess_only=preprocess_only, gpu=gpu)
        elif mode == 'image':
            output_filename = str(Path(output_dir) / (Path(audio_filename).stem + '.jpg'))
            from_audio_file_to_file(audio_filename, output_filename, textgrid_filename=textgrid_filename, checkpoint=checkpoint, font_filename=font_filename, prepocess_only=preprocess_only, gpu=gpu, mode='image')



def from_ppg_to_video(ppg, audio_filename, labels=ppgs.PHONEME_LIST):
    temp_filename = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    from_ppg_to_video_file(
        ppg=ppg,
        audio_filename=audio_filename,
        video_filename=temp_filename,
        labels=labels
    )
    with open(temp_filename, 'rb') as video_file:
        video = video_file.read()
    
    try:
        os.remove(temp_filename)
    except OSError:
        print(f'Failed to delete temporary file {temp_filename}')

    return video


def from_ppgs_to_videos(batched_ppgs, audio_filenames, labels=ppgs.PHONEME_LIST):
    videos = []
    for ppg, audio_filename in zip(batched_ppgs, audio_filenames):
        videos.append(from_ppg_to_video(ppg, audio_filename, labels=labels))
    return videos

if __name__ == '__main__':
    # audio_filenames = [f'data/cache/arctic/cmu_us_bdl_arctic/arctic_a000{i}.wav' for i in range(1,2)]
    # import yapecs
    # parser = yapecs.ArgumentParser()
    # parser.add_argument('files', nargs='+')
    # args = vars(parser.parse_args())
    # print(args)

    # from_files_to_files(args['files'], './tmp/', mode='image', checkpoint='runs/w2v2fb/00200000.pt', font_filename='/home/cameron/conda/envs/p/fonts/arial.ttf', preprocess_only=False, gpu=0)
    # from_file_to_file(
    #     './tmp/100038.wav',
    #     './tmp/100038-w2v2fc-pretrained.jpg',
    #     mode='image',
    #     checkpoint=None,
    #     textgrid_filename='./tmp/100038.textgrid',
    #     font_filename='/home/cameron/conda/envs/p/fonts/arial.ttf',
    #     gpu=None)

    # from_ppg_file_to_file(
    #     ppg_filename='./tmp/100038-W2V2FC-pretrained-ppg.pt', # red
    #     audio_filename='./tmp/100038.wav',
    #     output_filename='./tmp/comparison.jpg',
    #     # textgrid_filename='./tmp/100038.textgrid', # blue
    #     second_ppg_filename='./tmp/100038-w2v2ft-ppg.pt', # green
    #     font_filename='/home/cameron/conda/envs/p/fonts/arial.ttf'
    # )

    from_ppg_file_to_file(
        ppg_filename='./tmp/100038-w2v2ft-ppg.pt', # red
        audio_filename='./tmp/100038.wav',
        output_filename='./tmp/w2v2ft.jpg',
        textgrid_filename='./tmp/100038.textgrid', # blue
        font_filename='/home/cameron/conda/envs/p/fonts/arial.ttf'
    )
