import glob
import itertools
import json
import logging
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
import torchvision.transforms.functional as tv_f
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


_CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

_DALI_IMAGE_PIPELINES = {}


def _prepare_images_dali(images, device, args):
    """Decode and apply the canonical CLIP crop on GPU through DALI.

    The crop area (0.9--1.0), aspect ratio (3/4--4/3), ten attempts, bicubic
    resize, RGB conversion and CLIP normalization mirror image_transform().
    """
    try:
        from nvidia.dali import fn, pipeline_def, types
    except ImportError as exc:
        raise RuntimeError("--wds-dali-decode requires nvidia-dali-cuda120") from exc

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    batch_size = len(images)
    cache_key = (device_index, batch_size)
    pipe = _DALI_IMAGE_PIPELINES.get(cache_key)
    if pipe is None:
        target_size = getattr(args, "force_image_size", None) or 224
        if isinstance(target_size, (tuple, list)):
            target_h, target_w = int(target_size[-2]), int(target_size[-1])
        else:
            target_h = target_w = int(target_size)

        @pipeline_def(
            batch_size=batch_size,
            num_threads=4,
            device_id=device_index,
            # One training-loop batch is fed at a time. Synchronous execution
            # avoids DALI requesting a second external-source batch before the
            # Python loop has received it; model compute supplies the overlap.
            exec_pipelined=False,
            exec_async=False,
            seed=int(getattr(args, "seed", 0)) + device_index,
        )
        def dali_image_pipeline():
            encoded = fn.external_source(name="encoded", device="cpu")
            decoded = fn.decoders.image_random_crop(
                encoded,
                device="mixed",
                output_type=types.RGB,
                random_area=(0.9, 1.0),
                random_aspect_ratio=(0.75, 4.0 / 3.0),
                num_attempts=10,
            )
            resized = fn.resize(
                decoded,
                device="gpu",
                resize_x=target_w,
                resize_y=target_h,
                interp_type=types.INTERP_CUBIC,
                antialias=True,
            )
            return fn.crop_mirror_normalize(
                resized,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[value * 255.0 for value in _CLIP_IMAGE_MEAN],
                std=[value * 255.0 for value in _CLIP_IMAGE_STD],
            )

        pipe = dali_image_pipeline()
        pipe.build()
        _DALI_IMAGE_PIPELINES[cache_key] = pipe

    # DALI accepts a list of one-dimensional CPU uint8 tensors as an external
    # encoded-image batch. Its TensorListGPU exports directly through DLPack.
    pipe.feed_input("encoded", images)
    output, = pipe.run()
    # TensorListGPU exposes individual-sample capsules; collapse it to one
    # dense NHWC/NCHW TensorGPU before handing ownership to PyTorch.
    return torch.from_dlpack(output.as_tensor())


def prepare_images(images, device, input_dtype, args):
    if not getattr(args, "wds_gpu_decode", False):
        return images.to(device=device, dtype=input_dtype, non_blocking=True)
    if not isinstance(images, (list, tuple)):
        raise TypeError("GPU WebDataset decode expects a list of encoded JPEG tensors")
    if getattr(args, "wds_dali_decode", False):
        return _prepare_images_dali(images, device, args).to(dtype=input_dtype)
    decoded = tv_io.decode_jpeg(list(images), mode=tv_io.ImageReadMode.RGB, device=device)
    if getattr(args, "wds_gpu_random_resized_crop", False):
        target_size = getattr(args, "force_image_size", None) or 224
        if isinstance(target_size, (tuple, list)):
            output_size = [int(target_size[-2]), int(target_size[-1])]
        else:
            output_size = [int(target_size), int(target_size)]
        cropped = []
        for image in decoded:
            top, left, height, width = RandomResizedCrop.get_params(
                image, scale=(0.9, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            cropped.append(tv_f.resized_crop(
                image,
                top,
                left,
                height,
                width,
                output_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ))
        decoded = cropped
    elif len({tuple(image.shape) for image in decoded}) != 1:
        raise ValueError(
            "Variable-resolution JPEGs require --wds-gpu-random-resized-crop"
        )
    target_dtype = input_dtype or torch.float32
    images = torch.stack(decoded).to(dtype=target_dtype).div_(255.0)
    mean = torch.as_tensor(_CLIP_IMAGE_MEAN, device=device, dtype=target_dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(_CLIP_IMAGE_STD, device=device, dtype=target_dtype).view(1, 3, 1, 1)
    return images.sub_(mean).div_(std)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def all_ranks_finite(value):
    """Return True only when ``value`` is finite on every distributed rank.

    All ranks must make the same skip/step decision or DDP can deadlock on the
    following iteration. The one-element reduction protects long streaming
    runs from a single numerically bad batch.
    """
    finite = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    return bool(finite.item())


# Matches open_clip_train.main.LATEST_CHECKPOINT_NAME -- kept as a separate constant
# here to avoid a circular import (main.py imports this module, not the reverse).
LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def save_step_checkpoint(original_model, optimizer, scaler, epoch, step, args):
    """Save a resumable snapshot atomically."""
    if not is_master(args):
        return
    checkpoint_dict = {
        "epoch": epoch,
        "step": step,
        "next_step": step + 1,
        "name": args.name,
        "state_dict": original_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()

    step_path = os.path.join(args.checkpoint_path, f"step_{step}.pt")
    tmp_path = step_path + ".tmp"
    torch.save(checkpoint_dict, tmp_path)
    os.replace(tmp_path, step_path)
    logging.info(f"Saved step checkpoint {step_path}")

    if args.save_most_recent:
        tmp_latest = os.path.join(args.checkpoint_path, "tmp_latest.pt")
        latest_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
        torch.save(checkpoint_dict, tmp_latest)
        os.replace(tmp_latest, latest_path)

    max_keep = args.save_steps_max_keep
    if max_keep and max_keep > 0:
        step_files = glob.glob(os.path.join(args.checkpoint_path, "step_*.pt"))
        step_files = [p for p in step_files if re.search(r"step_(\d+)\.pt$", p)]
        step_files.sort(key=lambda p: int(re.search(r"step_(\d+)\.pt$", p).group(1)))
        for old_path in step_files[:-max_keep]:
            try:
                os.remove(old_path)
            except OSError:
                pass


def evaluate_quick(model, data, step, args, tb_writer=None):
    """Evaluate a subset of validation data."""
    val_info = data.get('val_quick', data.get('val'))
    if not is_master(args) or val_info is None:
        return {}
    device = torch.device(args.device)
    was_training = model.training
    model.eval()

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    max_samples = args.val_steps_max_samples

    all_image_features, all_text_features = [], []
    cumulative_loss = 0.0
    num_samples = 0
    logit_scale = None
    val_iter = iter(val_info.dataloader)
    try:
        with torch.inference_mode():
            for batch in val_iter:
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"].mean()
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())

                    batch_size = images.shape[0]
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss.item() * batch_size
                num_samples += batch_size
                if num_samples >= max_samples:
                    break
    finally:
        del val_iter

    if was_training:
        model.train()

    if num_samples == 0 or logit_scale is None:
        return {}

    val_metrics = get_clip_metrics(
        image_features=torch.cat(all_image_features),
        text_features=torch.cat(all_text_features),
        logit_scale=logit_scale.cpu(),
    )
    metrics = {**val_metrics, "clip_val_loss": cumulative_loss / num_samples, "num_samples": num_samples}

    logging.info(
        f"Quick Eval @ step {step} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val_quick/" + name: val for name, val in metrics.items()}
    if tb_writer is not None:
        for name, val in log_data.items():
            tb_writer.add_scalar(name, val, step)
    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        wandb.log(log_data, step=step)

    return metrics


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None, original_model=None, skip_batches=0):
    if original_model is None:
        original_model = model
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    profile_steps = int(os.environ.get("BIO_PROFILE_STEPS", "0") or 0)
    profile_warmup = int(os.environ.get("BIO_PROFILE_WARMUP", "5") or 5)
    profile_totals = {
        "input": 0.0,
        "forward_loss": 0.0,
        "loss_finite": 0.0,
        "backward": 0.0,
        "grad_clip": 0.0,
        "grad_finite": 0.0,
        "optimizer": 0.0,
    }
    profile_count = 0
    end = time.time()
    data_iter = iter(dataloader)
    if skip_batches > 0:
        if is_master(args):
            logging.info(f"Skipping {skip_batches} already-trained batches before resuming...")
        for skipped in range(1, skip_batches + 1):
            next(data_iter)
            if is_master(args) and (skipped % 1000 == 0 or skipped == skip_batches):
                logging.info(
                    f"Resume fast-forward: {skipped}/{skip_batches} batches "
                    f"({100.0 * skipped / skip_batches:.1f}%)"
                )
        # IterableDataset workers prefetch independently. Drain any placeholder
        # batches still queued after the globally counted skip, but never discard
        # a real image batch.
        residual_placeholders = 0
        first_real_batch = next(data_iter)
        def is_resume_placeholder(image_batch):
            if isinstance(image_batch, (list, tuple)):
                return bool(image_batch) and image_batch[0].numel() == len(b"__bio_resume_fast_forward__")
            return image_batch.shape[-2:] == (1, 1)

        while is_resume_placeholder(first_real_batch[0]):
            residual_placeholders += 1
            first_real_batch = next(data_iter)
        if is_master(args) and residual_placeholders:
            logging.info(
                f"Drained {residual_placeholders} residual resume placeholder batches."
            )
        data_iter = itertools.chain((first_real_batch,), data_iter)
    for i, batch in enumerate(data_iter, start=skip_batches):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        profiling = profile_steps > 0 and i < profile_warmup + profile_steps
        profile_record = profiling and i >= profile_warmup
        profile_last = time.perf_counter()

        def profile_mark(name):
            nonlocal profile_last
            if not profiling:
                return
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            if profile_record:
                profile_totals[name] += now - profile_last
            profile_last = now

        use_bio_metadata = args.bioloom_loss
        if use_bio_metadata:
            images, texts, bio_metadata = batch
            bio_metadata = bio_metadata.to(device=device, non_blocking=True)
        else:
            images, texts = batch
        images = prepare_images(images, device, input_dtype, args)
        token_length = int(getattr(args, "wds_token_length", 0) or 0)
        if token_length > 0 and texts.ndim == 2 and texts.shape[1] > token_length:
            texts = texts[:, :token_length]
        texts = texts.to(device=device, non_blocking=True)
        profile_mark("input")

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                if use_bio_metadata:
                    model_out["bio_metadata"] = bio_metadata
                if args.bioloom_loss:
                    loss.set_step(step)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            profile_mark("forward_loss")
            loss_is_finite = all_ranks_finite(total_loss)
            profile_mark("loss_finite")
            if not loss_is_finite:
                logging.error(
                    "Skipping non-finite forward loss at epoch=%d step=%d; components=%s",
                    epoch,
                    step,
                    {name: float(value.detach().float().cpu()) for name, value in losses.items()},
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            backward(total_loss, scaler)
            profile_mark("backward")
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                else:
                    grad_norm = None
                profile_mark("grad_clip")
                grad_is_finite = grad_norm is None or all_ranks_finite(grad_norm)
                profile_mark("grad_finite")
                if grad_is_finite:
                    scaler.step(optimizer)
                else:
                    logging.error(
                        "Skipping non-finite gradients at epoch=%d step=%d; grad_norm=%s amp_scale=%s",
                        epoch, step, float(grad_norm.detach().float().cpu()), scaler.get_scale(),
                    )
            scaler.update()
            profile_mark("optimizer")
        else:
            if args.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            else:
                grad_norm = None
            profile_mark("grad_clip")
            grad_is_finite = grad_norm is None or all_ranks_finite(grad_norm)
            profile_mark("grad_finite")
            if grad_is_finite:
                optimizer.step()
            else:
                logging.error(
                    "Skipping non-finite gradients at epoch=%d step=%d; grad_norm=%s",
                    epoch, step, float(grad_norm.detach().float().cpu()),
                )
            profile_mark("optimizer")

        if profile_record:
            profile_count += 1
            if profile_count == profile_steps and is_master(args):
                means = {name: value / profile_count for name, value in profile_totals.items()}
                measured = sum(means.values())
                logging.info(
                    "BIO_PROFILE mean over %d steps: %s measured_total=%.4fs",
                    profile_count,
                    " ".join(f"{name}={value:.4f}s" for name, value in means.items()),
                    measured,
                )

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        save_step_now = bool(
            args.save_steps_frequency and step > 0
            and step % args.save_steps_frequency == 0
        )
        eval_step_now = bool(
            args.val_steps_frequency and step > 0
            and step % args.val_steps_frequency == 0
        )
        if is_master(args):
            if save_step_now:
                save_step_checkpoint(original_model, optimizer, scaler, epoch, step, args)
            if eval_step_now:
                # Use eager mode to avoid recompiling the training graph.
                evaluate_quick(original_model, data, step, args, tb_writer=tb_writer)

        if args.distributed and (save_step_now or eval_step_now):
            # Wait for the rank handling validation and file writes.
            torch.distributed.barrier()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            wandb_log_now = (
                not args.wandb_log_frequency
                or step % args.wandb_log_frequency == 0
                or batch_count == num_batches_per_epoch
            )
            if args.wandb and wandb_log_now:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            # Bound the size of the retrieval similarity matrix.
            image_features_cat = torch.cat(all_image_features)
            text_features_cat = torch.cat(all_text_features)
            max_retrieval = args.val_retrieval_max_samples
            if max_retrieval and len(image_features_cat) > max_retrieval:
                g = torch.Generator().manual_seed(42)
                idx = torch.randperm(len(image_features_cat), generator=g)[:max_retrieval]
                image_features_cat = image_features_cat[idx]
                text_features_cat = text_features_cat[idx]
            val_metrics = get_clip_metrics(
                image_features=image_features_cat,
                text_features=text_features_cat,
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
