import signal
import os
import random


def get_n_running_proc(procs):
    statuses = [proc.poll() for proc in procs]
    n_proc = sum([1 for st in statuses if st is None])  # None from proc.poll() means that process is still running
    return n_proc


def get_n_gpu_proc(gpu):
    gpu_command = """nvidia-smi -g """ + str(gpu) + """ | awk '$2=="Processes:" {p=1} p && $3 > 0 {print $3}'"""
    output = os.popen(gpu_command).read()
    gpu_procs = [s for s in output.split('\n') if s not in ['GPU', 'PID', '', '0', 'running']]
    return len(gpu_procs)


def get_free_gpu():
    gpus = list(range(8))
    max_n_per_gpu = 1
    for gpu in gpus:
        if get_n_gpu_proc(gpu) < max_n_per_gpu:
            return gpu