import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger
import enlighten


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_dataset,
        val_dataset=None,
        loss_fn=None,
        device="cuda",
        batch_size=8,
        num_workers=4,
        distributed=False,
        save_dir="./checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCELoss(reduction="mean")
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.distributed = distributed
        self.rank = dist.get_rank() if distributed else 0
        self.world_size = dist.get_world_size() if distributed else 1
        self.epoch = 0
        self.best_metric = None
        self.train_sampler = (
            DistributedSampler(train_dataset) if distributed else None
        )
        self.val_sampler = (
            DistributedSampler(val_dataset)
            if (distributed and val_dataset is not None)
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=self.val_sampler,
            )
        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device] if torch.cuda.is_available() else None,
                output_device=self.rank,
                find_unused_parameters=True,
            )

    def train_one_epoch(self):
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        total_loss = 0
        total_samples = 0
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.train_loader),
                desc=f"[Epoch {self.epoch: 3d}] Training",
                unit="batch",
                color="green",
                leave=False,
            )
        for idx, batch in enumerate(self.train_loader):
            batch_data, batch_masks, _ = batch
            batch_data = batch_data.to(self.device)
            batch_masks = batch_masks.to(self.device)
            pred_logits = self.model(batch_data)
            loss = 0
            for pred_logit in pred_logits:
                loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Training] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
        if self.rank == 0:
            pbar.close()
        avg_loss = total_loss / total_samples
        # if self.rank == 0:
        #     logger.info(f"[Train][Epoch {self.epoch}] avg_loss={avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, metric_wrapper=None):
        if metric_wrapper is not None:
            metric_wrapper.reset()
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.val_loader),
                desc=f"[Epoch {self.epoch: 3d}] Evaluating",
                unit="batch",
                color="red",
                leave=False,
            )
        self.model.eval()
        if self.val_loader is None:
            return None
        total_loss = 0
        total_samples = 0
        for idx, batch in enumerate(self.val_loader):
            batch_data, batch_masks, _ = batch
            batch_data = batch_data.to(self.device)
            batch_masks = batch_masks.to(self.device)
            pred_logit = self.model(batch_data)[0]
            loss = 0
            loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Evaluating] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
            if metric_wrapper:
                metric_wrapper(pred_logit, batch_masks)
        avg_loss = total_loss / total_samples
        if self.rank == 0:
            logger.info(
                f"[Eval][Epoch {self.epoch}] avg_loss={avg_loss:.6f} metrics: {metric_wrapper}"
            )
        return avg_loss, metric_wrapper

    def save_checkpoint(self, save_path, extra=None):
        if self.rank != 0:
            return
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        state = {
            "epoch": self.epoch,
            "model_state_dict": self.model.module.state_dict()
            if self.distributed
            else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_metric": self.best_metric,
        }
        if extra:
            state.update(extra)
        torch.save(state, save_path)

    def load_checkpoint(self, path):
        map_location = (
            {"cuda:%d" % 0: "cuda:%d" % self.rank} if self.distributed else self.device
        )
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", None)
        return checkpoint
