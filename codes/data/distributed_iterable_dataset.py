# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Unimedvl Team

import random
import torch


class DistributedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, local_rank=0, world_size=1, num_workers=8):
        self.dataset_name = dataset_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.rng = random.Random()
        self.data_paths = None

    def get_data_paths(self, *args, **kwargs):
        raise NotImplementedError

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return

        if isinstance(self.data_paths[0], tuple):
            data_paths = sorted(self.data_paths, key=lambda x: (x[0], x[1]))
        elif isinstance(self.data_paths[0], str):
            data_paths = sorted(self.data_paths)
        else:
            raise ValueError(f"Unknown data_paths type: {type(self.data_paths[0])}")

        self.rng.seed(seed)
        self.rng.shuffle(data_paths)

        # Even distribution across ranks with remainder handling
        total = len(data_paths)
        base, rem = divmod(total, self.world_size)
        local_start = self.local_rank * base + min(self.local_rank, rem)
        local_end = local_start + base + (1 if self.local_rank < rem else 0)
        self.num_files_per_rank = max(0, local_end - local_start)
        self.data_paths_per_rank = data_paths[local_start:local_end]

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None

        info = torch.utils.data.get_worker_info()
        if info is None:
            # Single worker: Use all files assigned to the rank
            return self.data_paths_per_rank, 0

        worker_id = info.id
        all_paths = self.data_paths_per_rank
        n = len(all_paths)
        k = info.num_workers

        if n == 0:
            return [], worker_id

        # If fewer files than workers, assign at least one by modulo
        if n < k:
            return [all_paths[worker_id % n]], worker_id

        # Even distribution across workers with remainder handling
        base, rem = divmod(n, k)
        start = worker_id * base + min(worker_id, rem)
        end = start + base + (1 if worker_id < rem else 0)
        data_paths_per_worker = all_paths[start:end]

        # Fallback to avoid empty assignment
        if len(data_paths_per_worker) == 0:
            data_paths_per_worker = all_paths

        return data_paths_per_worker, worker_id

    def __iter__(self):
        raise NotImplementedError
