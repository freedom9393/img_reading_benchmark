import time
import functools


def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        num_of_imgs = func(*args, **kwargs)
        print(f"{kwargs['method']}: {time.time() - start_time}")
        return num_of_imgs
    return wrapper


def time_decorator_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        num_of_imgs = await func(*args, **kwargs)
        print(f"{kwargs.get('method', 'async')}: {time.time() - start_time}")
        return num_of_imgs
    return wrapper
