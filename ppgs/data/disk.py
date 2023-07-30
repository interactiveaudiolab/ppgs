from shutil import disk_usage

from ppgs import DATA_DIR, PRESERVED_DISK_SPACE_GB


def get_free_space_GB():
    available_bytes = disk_usage(DATA_DIR).free
    available_gigabytes = available_bytes / pow(1024, 3)
    return available_gigabytes

def stop_if_disk_full():
    if get_free_space_GB() < PRESERVED_DISK_SPACE_GB:
        raise OSError('Disk space is too low to continue operation')

def preserve_free_space(func: callable):
    def wrapper(*args, **kwargs):
        stop_if_disk_full
        func(*args, **kwargs)
    return wrapper
