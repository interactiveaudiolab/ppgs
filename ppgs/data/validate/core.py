import ppgs
import tqdm
from json import dumps
import pdb

def datasets(datasets, representation=None, partitions=None, debug=False):
    """Validate dataset caches have been established correctly"""

    if representation is None: representation = ppgs.REPRESENTATION

    for dataset in datasets:
        dataset_errors = {}
        for partition in partitions:
            dataset_object = ppgs.data.Dataset(dataset, partition, representation)
            iterator = tqdm.tqdm(
                range(0, len(dataset_object)),
                f"Validating dataset {dataset} with partition {partition} and representation {representation}",
                total=len(dataset_object),
                dynamic_ncols=True
            )
            for i in iterator:
                try:
                    #TODO add more checks
                    dataset_object.__getitem__(i)
                except Exception as e:
                    if debug:
                        pdb.set_trace()
                        dataset_object.__getitem__(i)
                    dataset_errors[dataset_object.stems[i]] = f"{e.__class__}: {e}"
        print(dumps(dataset_errors, indent=1))
