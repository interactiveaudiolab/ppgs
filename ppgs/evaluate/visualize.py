import ppgs

import tqdm
import torch
import torchaudio
from moviepy import editor as mpy
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

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

#TODO make scalefactor a parameter (currently is a hardcoded constant)
def from_logits_to_video_file(logits, audio_filename, video_filename, preprocess_only=False, labels=ppgs.PHONEME_LIST):
    """Takes logits of shape time,categories and creates a visualization"""
    audio = torchaudio.load(audio_filename)[0][0]
    audio_clip = mpy.AudioFileClip(audio_filename, fps=ppgs.SAMPLE_RATE)

    pixels = torch.nn.functional.softmax(logits, dim=1) * 255 #softmax to get nice distribution
    # pixels = torch.nn.functional.one_hot(torch.argmax(logits, dim=1), logits.shape[-1]) * 255
    pixels = torch.nn.functional.pad(pixels, (0, 0, pad, pad)) #pad so playhead is centered
    pixels = pixels.unsqueeze(-1).repeat(1,1,3) #unsqueeze and convert form greyscale to rgb

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
    
    if not hasattr(from_logits_to_video_file, 'overlay'):
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

        from_logits_to_video_file.overlay = overlay

    from_logits_to_video_file.overlay.set_duration(clip.duration)
    composite = mpy.CompositeVideoClip([clip, from_logits_to_video_file.overlay])
    clip.close()
    composite = composite.set_audio(audio_clip)

    #TODO add audio_codec as a parameter
    composite.write_videofile(video_filename, audio_codec='aac', logger=None)
    composite.close()


def from_file_to_file(audio_filename, video_filename, checkpoint=ppgs.DEFAULT_CHECKPOINT, prepocess_only=False, gpu=None):
    logits = ppgs.from_file(audio_filename, checkpoint=checkpoint, preprocess_only=prepocess_only, gpu=gpu).T
    from_logits_to_video_file(logits.cpu(), audio_filename, video_filename, preprocess_only=prepocess_only)


def from_files_to_files(audio_filenames, output_dir, checkpoint=ppgs.DEFAULT_CHECKPOINT, preprocess_only=False, gpu=None):
    iterator = tqdm.tqdm(
        audio_filenames,
        desc='Creating visualizations',
        total=len(audio_filenames),
        dynamic_ncols=True
    )
    for audio_filename in iterator:
        output_filename = str(Path(output_dir) / (Path(audio_filename).stem + '.mp4'))
        from_file_to_file(audio_filename, output_filename, checkpoint=checkpoint, prepocess_only=preprocess_only, gpu=gpu)


def from_logits_to_video(logits, audio_filename, labels=ppgs.PHONEME_LIST):
    temp_filename = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    from_logits_to_video_file(
        logits=logits,
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


def from_logits_to_videos(batched_logits, audio_filenames, labels=ppgs.PHONEME_LIST):
    videos = []
    for logits, audio_filename in zip(batched_logits, audio_filenames):
        videos.append(from_logits_to_video(logits, audio_filename, labels=labels))
    return videos

if __name__ == '__main__':
    # audio_filenames = [f'data/cache/arctic/cmu_us_bdl_arctic/arctic_a000{i}.wav' for i in range(1,2)]
    audio_filenames = ['tmp/arctic_a0001.wav']

    from_files_to_files(audio_filenames, './tmp/', checkpoint='runs/basemodel/00300000.pt', preprocess_only=False, gpu=0)

    
