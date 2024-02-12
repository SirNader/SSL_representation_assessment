import logging

import numpy as np
from kappadata import ModeWrapper, KDComposeCollator, SubsetWrapper
from torch.utils.data import DistributedSampler, DataLoader

from distributed.config import is_distributed, get_world_size
from providers.config_providers.noop_config_provider import NoopConfigProvider
from utils.num_worker_heuristic import get_num_workers


class DataContainer:
    def __init__(
            self,
            pin_memory,
            persistent_workers,
            num_workers=None,
            config_provider=None,
            generator=None,
            **datasets,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.config_provider = config_provider or NoopConfigProvider()
        self.generator = generator

        self.datasets = datasets
        self.persistent_loaders = {}

    @property
    def train(self):
        return self.get_dataset("train")

    def get_dataset(self, key, mode=None, max_size=None, return_ctx=False):
        dataset = self.datasets[key]
        if max_size is not None:
            dataset = SubsetWrapper(dataset, end_index=max_size)
        if mode is not None:
            dataset = ModeWrapper(dataset=dataset, mode=mode, return_ctx=return_ctx)
        return dataset

    def dataloader_from_key(
            self,
            dataset_key,
            mode,
            batch_size,
            shuffle=None,
            drop_last=None,
            return_ctx=False,
            max_size=None,
    ):
        # get dataset
        dataset = self.get_dataset(key=dataset_key, mode=mode, max_size=max_size, return_ctx=return_ctx)
        if max_size is not None:
            batch_size = min(batch_size, max_size)

        # create persistent loader
        if dataset_key == "train":
            shuffle = True if shuffle is None else shuffle
            drop_last = True if drop_last is None else drop_last
            loader = self.dataloader_from_dataset(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                is_train_dataset=True,
            )
        else:
            # only train loader uses shuffle and drop last -> it doesn't make sense to overwrite this in other datasets
            assert shuffle is None and drop_last is None
            loader = self.dataloader_from_dataset(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                is_train_dataset=False,
            )
        prefetch_factor_str = f" prefetch_factor={loader.prefetch_factor}" if loader.prefetch_factor != 2 else ""
        self.logger.info(
            f"created '{dataset_key}' dataloader (batch_size={batch_size} "
            f"num_workers={loader.num_workers} pin_memory={loader.pin_memory} "
            f"persistent_workers={loader.persistent_workers}{prefetch_factor_str})"
        )
        # add to wandb config
        self.config_provider.update({
            f"dataloader/{dataset_key}/num_workers": loader.num_workers,
            f"dataloader/{dataset_key}/pin_memory": loader.pin_memory,
            f"dataloader/{dataset_key}/persistent_workers": loader.persistent_workers,
            f"dataloader/{dataset_key}/prefetch_factor": loader.prefetch_factor
        })
        return loader

    def dataloader_from_dataset(
            self,
            dataset,
            batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=None,
            pin_memory=None,
            prefetch_factor=None,
            is_train_dataset=False,
    ):
        # create collator
        if dataset.collators is not None and len(dataset.collators) > 0:
            collator = KDComposeCollator(
                collators=dataset.collators,
                dataset_mode=dataset.mode,
                return_ctx=dataset.return_ctx,
            )
        else:
            collator = None

        # check dataset size with batch_size
        if drop_last:
            if len(dataset) < batch_size:
                self.logger.warning(
                    f"dataset is too small to drop_last ({len(bs)}<{batch_size}) "
                    f"-> using batch_size=len(dataset) and drop_last=False"
                )
                batch_size = len(dataset)
                drop_last = False
        elif len(dataset) < batch_size:
            self.logger.info(
                f"dataset smaller than batch_size ({len(dataset)}<{batch_size}) "
                f"-> using batch_size=len(dataset)"
            )
            batch_size = min(len(dataset), batch_size)
            # distributed edge cases not implemented yet
            if is_distributed():
                assert batch_size % get_world_size() == 0

        # distributed sampler if necessary
        if is_distributed():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        else:
            sampler = None

        num_workers = num_workers or self.num_workers or get_num_workers(dataset, batch_size)
        if is_train_dataset:
            pin_memory = pin_memory or self.pin_memory
        else:
            # TODO when a lot of test datasets are used we would want to use a high prefetch factor
            #  this would sometimes result in high memory consumptions which I think is because of the pin_memory
            pin_memory = False
        kwargs = {}
        if num_workers > 0:
            persistent_workers = self.persistent_workers
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor

            # if num_workers > n_batches -> scale down batchsize
            # this can occour in valid/test sets when a lot of workers are used with a large batch_size
            n_batches = int(np.ceil(len(dataset) / batch_size))
            if num_workers > n_batches:
                # if flexible_batch_size:
                #     # use less batch_size but more workers
                #     batch_size = max(1, int(batch_size / 4))
                #     n_batches = int(np.ceil(len(dataset) / batch_size))
                #     num_workers = min(n_batches, num_workers)
                # else:
                # limit num_workers to n_batches (e.g. train dataset requires the full batch_size)
                num_workers = n_batches

            # TODO when a lot of eval datasets are used the process crashes
            if not is_train_dataset:
                num_workers = 4
                kwargs["prefetch_factor"] = kwargs.get("prefetch_factor", 2) * 2
        else:
            persistent_workers = False
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collator,
            generator=self.generator,
            **kwargs,
        )

    def dispose(self):
        for dataset in self.datasets.values():
            dataset.dispose()
