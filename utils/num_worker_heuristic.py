import os

from datasets.sensonic_png_indexed import SensonicPngIndexed
from distributed.config import is_slurm_run


# noinspection PyUnusedLocal
def get_num_workers(dataset, batch_size):
    cpu_count = _get_cpu_count()
    if cpu_count <= 16:
        # don't bother on dev machines
        return 0

    # get number of devices per node (srun nvidia-smi shows all devices not only the ones assigned for the srun task)
    # (if no GPU is available this returns "")
    devices_on_node = len(os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read().strip().split("\n"))
    if devices_on_node == 0:
        devices_on_node = 1

    # divide cpus among devices
    if is_slurm_run():
        # slurm already divides cpus among tasks -> assert that
        tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
        # currently only 1 GPU per task is supported
        assert devices_on_node == tasks_per_node
        assert cpu_count == cpus_per_task
        # use 75% of slurm workers for dataloading
        # 16worker MAE-B 512bs/A100 -> 0.05 data time
        # 24worker MAE-B 512bs/A100 -> 0.00 data time
        return int(cpu_count * 0.75)
    else:
        cpu_count = int(cpu_count / devices_on_node)

    if isinstance(dataset.root_dataset, SensonicPngIndexed):
        return cpu_count * 2

    # use less than cpu_count (too many workers often run into errors)
    # - "OSError: [Errno 24] Too many open files"
    # - "RuntimeError: Too many open files. Communication with the workers is no longer possible. ..."
    # max_workers = int(cpu_count / 2)
    max_workers = int(cpu_count - 1)
    return max_workers

    # scale num_workers with batchsize
    # num_workers = int(batch_size / 32)
    # multi_view_wrappers = [wt for wt in dataset.all_wrappers if isinstance(wt, MultiViewWrapper)]
    # if len(multi_view_wrappers) > 0:
    #     assert len(multi_view_wrappers) == 1
    #     n_views = multi_view_wrappers[0].n_views
    #     num_workers *= n_views
    # max_workers = max(4, min(max_workers, num_workers))

    # if isinstance(dataset.root_dataset, ImageNet):
    #     num_workers = int(batch_size / 32)
    #     multi_view_wrappers = [wt for wt in dataset.all_wrappers if isinstance(wt, MultiViewWrapper)]
    #     if len(multi_view_wrappers) > 0:
    #         assert len(multi_view_wrappers) == 1
    #         n_views = multi_view_wrappers[0].n_views
    #         num_workers *= n_views
    #     return max(4, min(max_workers, num_workers))


def _get_cpu_count():
    if os.name == "nt":
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        return cpu_count
    else:
        return len(os.sched_getaffinity(0))
