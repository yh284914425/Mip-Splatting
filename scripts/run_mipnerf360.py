# 原始 mipnerf-360 中的单尺度训练和单尺度测试

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

excluded_gpus = set([])

output_dir = "benchmark_360v2_ours"

dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s 360_v2/{scene} -m {output_dir}/{scene} --eval -r {factor} --port {6009+int(gpu)} --kernel_size 0.1"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --data_device cpu --skip_train"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene} -r {factor}"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    return True


def worker(gpu, scene, factor):
    print(f"在 GPU {gpu} 上使用场景 {scene} 启动任务\n")
    train_scene(gpu, scene, factor)
    print(f"在 GPU {gpu} 上使用场景 {scene} 完成任务\n")
    # 这个 worker 函数启动一个任务并在完成后返回。
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # 已计划工作但可能尚未激活的 GPU

    while jobs or future_to_job:
        # 获取可用 GPU 列表，不包括已预留的 GPU。
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        # 在可用 GPU 上启动新任务
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # 将任务解包为 worker 的参数
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # 预留此 GPU，直到任务开始处理

        # 检查已完成的任务，并将其从正在运行的任务列表中删除。
        # 同时，释放它们正在使用的 GPU。
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # 删除与已完成的 future 关联的任务
            gpu = job[0]  # GPU 是每个任务元组中的第一个元素
            reserved_gpus.discard(gpu)  # 释放此 GPU
            print(f"任务 {job} 已完成，正在释放 GPU {gpu}")
        # （可选）您可能希望在此处引入一个小的延迟，以防止在没有可用 GPU 时此循环过快地旋转。
        time.sleep(5)
        
    print("所有任务都已处理完毕。")


# 使用 ThreadPoolExecutor 管理线程池
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

