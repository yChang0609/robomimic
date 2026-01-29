import os
import torch
import threading
import queue
import time
from copy import deepcopy

class SaveManager:
    def __init__(self):
        self.regular_queue = queue.Queue()
        self.temp_slot = None
        self.temp_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        while not (self.stop_event.is_set() and self.regular_queue.empty() and self.temp_slot is None):
            task = None
            if not self.regular_queue.empty(): 
                task = self.regular_queue.get()
                is_temp = False
            else:
                with self.temp_lock:
                    if self.temp_slot is not None:
                        task = self.temp_slot
                        self.temp_slot = None 
                        is_temp = True
            
            if task is None:
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)
                continue

            data, path = task
            try:
                tmp_path = path + ".tmp"
                torch.save(data, tmp_path)
                os.replace(tmp_path, path)
                # type_str = "Temp" if is_temp else "Regular"
                # print(f"\n[{type_str} Save Done] {os.path.basename(path)}")
            except Exception as e:
                print(f"\n[Save Error] {path}: {e}")
            
            if not is_temp:
                self.regular_queue.task_done()

    def stop(self):
        print("\n[SaveManager] Waiting for remaining save jobs to finish...")
        self.stop_event.set()
        self.worker_thread.join()
        print("[SaveManager] All saves completed. Thread exited.")

    def __enter__(self):
        return self 
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def submit_regular(self, data, path):
        self.regular_queue.put((data, path))

    def submit_temp(self, data, path):
        with self.temp_lock:
            self.temp_slot = (data, path)