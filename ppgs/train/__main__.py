import argparse
import shutil
import os
from pathlib import Path

import ppgs
from ppgs.notify import notify_on_finish

###############################################################################
# Entry point
###############################################################################

@notify_on_finish('training')
def main(config, dataset, eval_only=False):
    # Create output directory
    directory = ppgs.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    ppgs.train.run(
        dataset,
        directory,
        directory,
        directory,
        eval_only)

    # Evaluate
    # ppgs.evaluate.datasets([dataset], directory, gpus)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        default=ppgs.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default='charsiu',
        help='The dataset to train on')
    # parser.add_argument(
    #     '--gpus', '--gpu',
    #     dest='gpus',
    #     type=int,
    #     nargs='+',
    #     help='The gpus to run training on')
    # parser.add_argument(
    #     '--no-cache',
    #     action='store_true',
    #     help='Do not use cache, do preprocessing on the fly'
    # )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation. For debugging purposes.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    print('CUDA_VISIBLE_DEVICES =', os.environ['CUDA_VISIBLE_DEVICES'])
    main(**vars(parse_args()))
