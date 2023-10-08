import bdb
import os
import time

import ppgs


###############################################################################
# Send job notifications
###############################################################################


def notify_on_finish(
    description: str,
    track_time: bool = True,
    notify_on_fail: bool = True):
    """Context manager for sending job notifications"""
    def wrapper(func: callable):
        def _wrapper(*args, **kwargs):
            # Start time
            if track_time:
                start_time = time.time()

            # Run callable
            try:
                func(*args, **kwargs)
            except Exception as exception:

                # Ignore pdb exceptions
                if isinstance(exception, bdb.BdbQuit):
                    return

                # Report failure
                if notify_on_fail:
                    message = (
                        f'task "{description}" failed with '
                        'exception: {exception.__class__}')
                    push(message)

                raise exception

            # End time
            if track_time:
                end_time = time.time()

            # Report success
            if track_time:
                message = (
                    f'Task "{description}" finished in '
                    f'{round(end_time - start_time)} seconds')
            else:
                message = f'Task "{description}" finished'
            push(message)

        return _wrapper
    return wrapper


###############################################################################
# Utilities
###############################################################################


def push(message: str):
    """send a push notification to all of the configured services"""
    service = os.getenv('PPGS_NOTIFICATION_URL', default=None)
    if service is not None:
        if not hasattr(push, 'messenger'):
            import apprise
            push.messenger = apprise.Apprise()
            push.messenger.add(service)
        push.messenger.notify(message)
