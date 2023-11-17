import os

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import train_utils, Logs
from model import model_LSTM
import torch.nn.functional as F
from utils.extract_flows import mkdir_p
import json
import argparse
import time
from model.network import set_parameter_requires_grad
from torch.utils.tensorboard import SummaryWriter
import options


def ddp_setup(local_rank):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # rank 0 process
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
    # nccl：NVIDIA Collective Communication Library
    # 分布式情况下的，gpus 间通信
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    print(f"[init] == local rank: {int(os.environ['LOCAL_RANK'])} ==")


class Trainer:
    def __init__(self,
                 local_rank: int,
                 model: torch.nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion,
                 exp_name,
                 cfg
                 ) -> None:
        self.gpu_id = local_rank
        self.model = model.to(self.gpu_id)
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = DDP(model, device_ids=[
                         self.gpu_id], find_unused_parameters=True)
        self.cfg = cfg
        self.exp_name = exp_name
        if local_rank == 0:
            print(f"The log of this experiment is saved in “{exp_name}”")
            self.writer_t = SummaryWriter(
                "{}/{}/train".format(self.cfg.logs, exp_name))
            self.writer_v = SummaryWriter(
                "{}/{}/val".format(self.cfg.logs, exp_name))
            mkdir_p(self.cfg.model_save_dir)
            mkdir_p(os.path.join(self.cfg.model_save_dir, exp_name))
            self.checkpoint = {
                'net': None,
                'optimizer': None,
                'epoch': None
            }
            self.model_json = self.cfg.__dict__
            self.model_json_path = os.path.join(
                self.cfg.logs, exp_name + ".json")
            if os.path.exists(self.model_json_path):
                with open(self.model_json_path, 'r') as f:
                    logs = json.load(f)
                    self.model_json["loss"] = logs["loss"]

    def _train_run_batch(self, input, label):
        self.optimizer.zero_grad()
        output = self.model(*input)
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        acc = (1 - (output.detach() - label.detach()).abs()
               ).sum(0) / label.size(0)
        MACC = acc.sum() / 5

        return MACC, loss.item()

    def _val_run_batch(self, input, label):
        with torch.no_grad():
            output = self.model(*input)
            loss = self.criterion(output, label)
            acc = (1 - (output.detach() - label.detach()).abs()
                   ).sum(0) / label.size(0)
            MACC = acc.sum() / 5

        return MACC, loss.item()

    def _run_epoch(self, epoch: int, is_train: bool = True):
        total_loss = total_MACC = 0
        lens = len(self.train_dataloader) if is_train else len(
            self.val_dataloader)

        if is_train:
            self.train_dataloader.sampler.set_epoch(epoch)
            start_time = time.time()
            for idx, (xs, ys) in enumerate(self.train_dataloader):
                xs = (x.to(self.gpu_id) for x in xs)
                ys = ys.to(self.gpu_id)
                MACC, loss = self._train_run_batch(xs, ys)
                total_MACC += MACC
                total_loss += loss
                end_time = time.time()
                if self.gpu_id == 0 and ((idx + 1) % 10 == 0 or (idx + 1) == lens):
                    print(
                        "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | MACC: {:6.3f}% | time: {:2.1f}s/it".format(
                            idx + 1,
                            lens,
                            epoch,
                            self.cfg.epochs,
                            total_loss / (idx + 1),
                            100.0 * total_MACC / (idx + 1),
                            (end_time - start_time) / (idx + 1)
                        )
                    )
                    writer = self.writer_t if is_train else self.writer_v
                    writer.add_scalar("loss", total_loss /
                                      (idx + 1), epoch*lens + idx + 1)
                    writer.add_scalar("MACC", 100.0 * total_MACC /
                                      (idx + 1), epoch*lens + idx + 1)
                    writer.add_scalar(
                        "Lr", self.optimizer.param_groups[0]['lr'], epoch*lens + idx + 1)
        else:
            self.val_dataloader.sampler.set_epoch(epoch)
            start_time = time.time()
            for idx, (xs, ys) in enumerate(self.val_dataloader):
                xs = (x.to(self.gpu_id) for x in xs)
                ys = ys.to(self.gpu_id)
                MACC, loss = self._val_run_batch(xs, ys)
                total_MACC += MACC
                total_loss += loss
                end_time = time.time()
                if self.gpu_id == 0 and ((idx + 1) % 20 == 0 or (idx + 1) == lens):
                    print(
                        "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | MACC: {:6.3f}% | time: {:2.1f}s/it".format(
                            idx + 1,
                            lens,
                            epoch,
                            self.cfg.epochs,
                            total_loss / (idx + 1),
                            100.0 * total_MACC / (idx + 1),
                            (end_time - start_time) / (idx + 1)
                        )
                    )
                    writer = self.writer_t if is_train else self.writer_v
                    writer.add_scalar("loss", total_loss /
                                      (idx + 1), epoch*lens + idx + 1)
                    writer.add_scalar("MACC", 100.0 * total_MACC /
                                      (idx + 1), epoch*lens + idx + 1)
                    writer.add_scalar(
                        "Lr", self.optimizer.param_groups[0]['lr'], epoch*lens + idx + 1)
        if self.gpu_id == 0:
            self.model_json = train_utils.update_json(model_json=self.model_json, model=self.model, optimizer=self.optimizer, epoch=epoch, loss=total_loss / lens, MACC=total_MACC /
                                                      lens, checkpoint=self.checkpoint, best_model_save_dir=self.cfg.best_model_save_dir, model_save_dir=self.cfg.model_save_dir, exp_name=self.exp_name, is_train=is_train)

            with open(self.model_json_path, 'w') as f:
                json.dump(self.model_json, f, indent=4, ensure_ascii=False)

    def train(self, start_epoch: int, max_epoch: int):
        for epoch in range(start_epoch, max_epoch):
            if epoch == 2 and args.backbone.startswith("resnet"):
                model1 = self.model.module
                set_parameter_requires_grad(model=model1, freezing=False)
            self._run_epoch(epoch)
            if epoch % 3 == 0:
                self._run_epoch(epoch=epoch, is_train=False)


def main(args: argparse):
    local_rank = int(int(os.environ['LOCAL_RANK']))
    ddp_setup(local_rank=local_rank)
    exp_name = Logs.get_exp_name(args=args)

    train_dataloader = train_utils.ddp_get_dataloader(cfg=args,
                                                      is_train=True)
    val_dataloader = train_utils.ddp_get_dataloader(cfg=args,
                                                    is_train=False)

    model = model_LSTM.BIO_MODEL_LSTM(arg=args)
    model, start_epoch, opt_para = train_utils.ddp_get_model(
        cfg=args, model=model, exp_name=exp_name)
    if start_epoch is None:
        start_epoch = 0

    optimizer = train_utils.get_opt(cfg=args, model=model, opt_para=opt_para)
    criterion = nn.MSELoss()

    trainer = Trainer(local_rank=local_rank, model=model,
                      optimizer=optimizer,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      criterion=criterion, exp_name=exp_name, cfg=args)
    trainer.train(start_epoch=start_epoch, max_epoch=args.epochs)

    dist.destroy_process_group()


if __name__ == "__main__":
    args = options.get_args()
    main(args=args)
