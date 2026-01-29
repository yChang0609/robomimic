import time
import torch
from datetime import timedelta
from contextlib import contextmanager

class TrainingTimer:
    def __init__(self):
        self.start_time = time.perf_counter()
    
    def get_elapsed_time(self):
        elapsed = time.perf_counter() - self.start_time
        return str(timedelta(seconds=int(elapsed)))
    
@contextmanager
def section_timer(name="Code Block Name"):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    yield
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end = time.perf_counter()
    print(f"[{name}] cost: {end - start:.6f} second")