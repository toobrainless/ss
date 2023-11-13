from pathlib import Path
from random import shuffle

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.utils import MetricTracker, inf_loop, normalize_loud

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
        keyboard_interrupt_save=True,
    ):
        super().__init__(
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            keyboard_interrupt_save,
        )
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.lr_scheduler = lr_scheduler
        self.log_step = self.config["trainer"].get("log_step", 50)

        shared_metrics = [
            m for m_list in self.metrics.get("shared", []) for m in m_list.get_metrics()
        ] + ["loss"]
        self.train_metrics = MetricTracker(
            "grad norm",
            *(
                [
                    m
                    for m_list in self.metrics.get("train", [])
                    for m in m_list.get_metrics()
                ]
                + shared_metrics
            ),
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *(
                [
                    m
                    for m_list in self.metrics.get("evaluation", [])
                    for m in m_list.get_metrics()
                ]
                + shared_metrics
            ),
            writer=self.writer,
        )
        self.accumulation_steps = self.config["trainer"].get("accumulation_steps", 1)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in [
            "ref_audio",
            "mix_audio",
            "target_audio",
            "ref_target",
            "ref_length",
        ]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    # TODO implemented batch accumulation but metric calculated only on the last batch
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # self.logger.info(f"start {epoch} epoch")
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx in tqdm(range(self.len_epoch), desc="train"):
            try:
                for batch_num, batch in enumerate(self.train_dataloader):
                    batch = self.process_batch(
                        batch_num,
                        batch,
                        is_train=True,
                        metrics=self.train_metrics,
                    )
                    if batch_num + 1 >= self.accumulation_steps:
                        break
            except RuntimeError as e:
                if ("out of memory" in str(e)) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler is not None:
                    if isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        _lr = self.optimizer.param_groups[0]["lr"]
                    else:
                        _lr = self.lr_scheduler.get_last_lr()[0]

                    self.writer.add_scalar("learning rate", _lr)
                self._log_predictions(**batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            # self.logger.info(f"finish {batch_idx} batch")
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(log["loss"])

        # self.logger.info(f"finish {epoch} epoch")
        return log

    def process_batch(self, batch_num, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train and batch_num % self.accumulation_steps == 0:
            self.optimizer.zero_grad()
        outputs = self.model(
            batch["mix_audio"], batch["ref_audio"], batch["ref_length"]
        )
        if isinstance(outputs, dict):
            batch.update(outputs)
        else:
            batch["logits"] = outputs
        batch["loss"] = self.criterion(
            batch["estimate_short"],
            batch["estimate_middle"],
            batch["estimate_long"],
            batch["target_audio"],
            batch["ref_target"],
            batch["logits"],
            include_ce=True,
        )
        # print(f"{batch.keys()=}")
        if (batch_num + 1) % self.accumulation_steps == 0:
            if is_train:
                batch["loss"].backward()
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None and not isinstance(
                    self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.lr_scheduler.step()
            # TODO fix this dumb shit
            with torch.no_grad():
                metrics.update("loss", -batch["loss"].item())
                for met in self.metrics["shared"]:
                    metrics.update(met.name, met(**batch))
                if is_train:
                    for met in self.metrics["train"]:
                        metrics.update(met.name, met(**batch))
                else:
                    for met in self.metrics["evaluation"]:
                        metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    self.accumulation_steps - 1,
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
        self,
        ref_audio,
        ref_length,
        ref_path,
        target_audio,
        target_length,
        target_path,
        mix_audio,
        mix_length,
        mix_path,
        ref_speaker_id,
        ref_target,
        estimate_short,
        estimate_middle,
        estimate_long,
        examples_to_log=1,
        *args,
        **kwargs,
    ):
        # TODO: implement logging of beam search results
        if self.writer is None:
            return
        tuples = list(
            zip(
                ref_audio,
                ref_length,
                ref_path,
                target_audio,
                target_length,
                target_path,
                mix_audio,
                mix_length,
                mix_path,
                ref_speaker_id,
                ref_target,
                estimate_short,
                estimate_middle,
                estimate_long,
            )
        )
        shuffle(tuples)
        rows = {}
        for (
            ref_audio,
            ref_length,
            ref_path,
            target_audio,
            target_length,
            target_path,
            mix_audio,
            mix_length,
            mix_path,
            ref_speaker_id,
            ref_target,
            estimate_short,
            estimate_middle,
            estimate_long,
        ) in tuples[:examples_to_log]:
            triplet_id = Path(mix_path).stem.split("-")[0]
            rows[triplet_id] = {}
            for name, audio, length in zip(
                ["ref_audio", "target_audio", "mix_audio"],
                [ref_audio, target_audio, mix_audio],
                [ref_length, target_length, mix_length],
            ):
                rows[triplet_id][name] = self.writer.wandb.Audio(
                    normalize_loud(audio[:length].detach().cpu().numpy()),
                    sample_rate=self.config["preprocessing"]["sr"],
                )
            rows[triplet_id]["ref_speaker_id"] = ref_speaker_id
            rows[triplet_id]["ref_target"] = ref_target
            rows[triplet_id]["estimate_short"] = self.writer.wandb.Audio(
                normalize_loud(estimate_short.detach().cpu().numpy()[:mix_length]),
                sample_rate=self.config["preprocessing"]["sr"],
            )
            rows[triplet_id]["estimate_middle"] = self.writer.wandb.Audio(
                normalize_loud(estimate_middle.detach().cpu().numpy()[:mix_length]),
                sample_rate=self.config["preprocessing"]["sr"],
            )
            rows[triplet_id]["estimate_long"] = self.writer.wandb.Audio(
                normalize_loud(estimate_long.detach().cpu().numpy()[:mix_length]),
                sample_rate=self.config["preprocessing"]["sr"],
            )
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
