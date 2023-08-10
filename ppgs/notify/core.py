import bdb
import time

import apprise

from ppgs import NOTIFICATION_SERVICES

messenger = apprise.Apprise()
for service in NOTIFICATION_SERVICES:
    messenger.add(service)

def push(message: str):
    """send a push notification to all of the configured services"""
    messenger.notify(message)

def notify_on_finish(description: str, track_time: bool = True, notify_on_fail: bool = True):
    def wrapper(func: callable):
        def _wrapper(*args, **kwargs):
            if track_time:
                start_time = time.time()
            print('starting notified task:', description)
            try:
                func(*args, **kwargs)
            except Exception as exception:
                if isinstance(exception, bdb.BdbQuit):
                    return
                if notify_on_fail:
                    message = f'task "{description}" failed with exception: {exception.__class__}'
                    push(message)
                raise exception
            if track_time:
                end_time = time.time()
            print('task finished, sending notifications')
            if track_time:
                message = f'Task "{description}" finished in {round(end_time-start_time)} seconds'
            else:
                message = f'Task "{description}" finished'
            push(message)
        return _wrapper
    return wrapper

if __name__ == '__main__':
    @notify_on_finish('testing', True)
    def __test_task(seconds: int):
        time.sleep(seconds)
    __test_task(10)
