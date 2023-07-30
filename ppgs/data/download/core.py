import ppgs
import ppgs.data.purge
from ppgs.notify import notify_on_finish

from . import datasets as dataset_objects


@notify_on_finish('download')
def datasets(datasets, format_only, purge_sources):
    """Downloads the datasets passed in"""
    datasets = [dataset.lower() for dataset in datasets]
    for dataset in datasets:
        if hasattr(dataset_objects, dataset):
            dataset_object = getattr(dataset_objects, dataset)
            if not format_only:
                dataset_object.download()
                dataset_object.format()
            else:
                dataset_object.format()
            if purge_sources:
                ppgs.data.purge.datasets(datasets=[dataset], kinds=['sources'])