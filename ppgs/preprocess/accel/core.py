import time
import multiprocessing as mp
import tqdm
from ppgs.data import stop_if_disk_full
from .utils import *

#TODO allow control of batch size*
def multiprocessed_preprocess(dataset, output_dir, features, num_workers=0, gpu=None):
    #half the threads for loading...
    dataloader = loader(dataset, num_workers=num_workers//2)
    feature_processors = [ppgs.REPRESENTATION_MAP[f] for f in features]
    iterator = tqdm.tqdm(
        dataloader,
        desc=f'preprocessing {features} for dataset {dataset}',
        total=len(dataloader),
        dynamic_ncols=True
    )
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    wasted_time = 0
    #... and the other half for saving
    with mp.get_context('spawn').Pool(num_workers//2) as pool:
        with torch.inference_mode():
            for audios, audio_files, lengths in iterator:
                audios = audios.to(device)
                lengths = lengths.to(device)
                for feature, feature_processor in zip(features, feature_processors):
                    # torch.cuda.empty_cache()
                    # print(torch.cuda.memory_summary(gpu, abbreviated=True))
                    outputs = feature_processor.from_audios(audios, lengths, gpu=gpu).cpu()
                    assert str(outputs.device) == 'cpu', f'"{outputs.device}"'
                    new_lengths = [length // ppgs.HOPSIZE for length in lengths]
                    filenames = [output_dir / f'{audio_file.stem}-{feature}.pt' for audio_file in audio_files]
                    pool.starmap_async(save_masked, zip(outputs, filenames, new_lengths))
                    while pool._taskqueue.qsize() > 256:
                        time.sleep(1)
                        wasted_time += 1
            stop_if_disk_full()
        pool.close()
        pool.join()
        print("~total wasted time: ", wasted_time)


def multiprocessed_process(
    dataset,
    output_dir,
    from_features,
    save_from_features=False,
    num_workers=0,
    gpu=None):
    #half the threads for loading...
    dataloader = loader(dataset, num_workers=num_workers//2)
    feature_processors = [ppgs.REPRESENTATION_MAP[f] for f in from_features]
    iterator = tqdm.tqdm(
        dataloader,
        desc=f'processing {from_features} for dataset {dataset}',
        total=len(dataloader),
        dynamic_ncols=True
    )
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    wasted_time = 0
    #... and the other half for saving
    with mp.get_context('spawn').Pool(num_workers//2) as pool:
        with torch.inference_mode(), torch.autocast(device_type='cuda'): #TODO fix for CPU
            for audios, audio_files, lengths in iterator:
                audios = audios.to(device)
                lengths = lengths.to(device)
                for feature, feature_processor in zip(from_features, feature_processors):
                    # torch.cuda.empty_cache()
                    # print(torch.cuda.memory_summary(gpu, abbreviated=True))
                    outputs = feature_processor.from_audios(audios, lengths, gpu=gpu)
                    new_lengths = lengths // ppgs.HOPSIZE
                    ppg_outputs = feature_processor.from_features(outputs, new_lengths, gpu=gpu)
                    if save_from_features:
                        filenames = [output_dir / f'{audio_file.stem}-{feature}.pt' for audio_file in audio_files]
                        pool.starmap_async(save_masked, zip(outputs.cpu(), filenames, new_lengths))
                    filenames = [output_dir / f'{audio_file.stem}-{feature}-ppg.pt' for audio_file in audio_files]
                    pool.starmap_async(save_masked, zip(ppg_outputs.cpu(), filenames, new_lengths))
                    while pool._taskqueue.qsize() > 256:
                        time.sleep(1)
                        wasted_time += 1
                stop_if_disk_full()
        pool.close()
        pool.join()
        print("~total wasted time: ", wasted_time)