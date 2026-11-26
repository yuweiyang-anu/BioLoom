import ast
import io
import json
import logging
import math
import os
import random
import hashlib
import sys
import braceexpand
from collections import Counter, deque
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.io as tv_io
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def decode_image_tensor(data):
    """Decode JPEG/PNG bytes directly to CHW uint8 without a PIL round trip."""
    encoded = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return tv_io.decode_image(encoded, mode=tv_io.ImageReadMode.RGB)


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            # Flush the final sample at each shard boundary.
            if not filesample:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = None
                continue
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            # Duplicate suffixes indicate a new sample.
            if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
        except Exception as exn:
            if handler is not None and handler(exn):
                continue
            else:
                logging.warning(f"group_by_keys_nothrow: skipping malformed sample ({repr(exn)})")
                continue
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class FigureGroupShuffle(wds.PipelineStage):
    """Shuffle small figure chunks while retaining explicit sibling pairs.

    Input samples must be contiguous by ``figure_id`` inside each source
    shard. Large figures are split into bounded chunks so no batch can be
    dominated by one figure. The buffer is measured in chunks, not samples.
    """

    def __init__(self, bufsize=512, max_samples=2, seed=0, epoch=-1,
                 type_aware=False, balance_power=0.0, balance_window=0,
                 balance_candidates=0, rotate_panels=False):
        if bufsize <= 0:
            raise ValueError("figure-group shuffle bufsize must be positive")
        if max_samples <= 0:
            raise ValueError("figure-group max_samples must be positive")
        self.bufsize = bufsize
        self.max_samples = max_samples
        self.seed = seed
        self.epoch = epoch
        self.type_aware = type_aware
        if balance_power < 0:
            raise ValueError("figure-group balance_power must be non-negative")
        if balance_window < 0:
            raise ValueError("figure-group balance_window must be non-negative")
        self.balance_power = balance_power
        self.balance_window = balance_window
        self.rotate_panels = rotate_panels
        self._active_epoch = 0
        if balance_candidates < 0:
            raise ValueError("figure-group balance_candidates must be non-negative")
        self.balance_candidates = balance_candidates

    @staticmethod
    def _metadata(sample):
        cached = sample.get("__bio_metadata_cache__")
        if isinstance(cached, dict):
            return cached
        payload = sample.get("json")
        try:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            metadata = json.loads(payload) if isinstance(payload, str) else payload
            metadata = metadata if isinstance(metadata, dict) else {}
            sample["__bio_metadata_cache__"] = metadata
            return metadata
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _figure_id(sample):
        metadata = FigureGroupShuffle._metadata(sample)
        figure_id = metadata.get("figure_id") if isinstance(metadata, dict) else None
        # Invalid metadata is still kept, but never grouped with another sample.
        return figure_id or f"__sample__:{sample.get('__url__')}:{sample.get('__key__')}"

    def _chunks(self, group):
        """Yield bounded, lossless chunks with an informative first bundle.

        Without type-aware grouping, samples are chunked in their input order.
        Type-aware grouping expects hierarchical shards, where every
        figure is stored as whole view followed by its panels. For a mixed
        figure the first bundle is whole + non-chart panel + chart panel;
        remaining panels are emitted exactly once in bounded chunks.
        """
        if not self.type_aware:
            for start in range(0, len(group), self.max_samples):
                yield group[start:start + self.max_samples]
            return

        remaining = list(group)
        if self.rotate_panels and remaining:
            # Keep all samples, but rotate which panels accompany the whole
            # view so low-index panels are not privileged in every epoch.
            whole = [sample for sample in remaining
                     if self._metadata(sample).get("view_type") == "whole"]
            non_chart = [sample for sample in remaining
                         if self._metadata(sample).get("view_type") == "panel"
                         and self._metadata(sample).get("content_type") != "chart"]
            chart = [sample for sample in remaining
                     if self._metadata(sample).get("view_type") == "panel"
                     and self._metadata(sample).get("content_type") == "chart"]
            selected_ids = {id(sample) for sample in whole + non_chart + chart}
            other = [sample for sample in remaining if id(sample) not in selected_ids]
            figure_id = self._figure_id(remaining[0])

            def rotate(items, namespace):
                if len(items) < 2:
                    return items
                digest = hashlib.blake2b(
                    f"{figure_id}:{namespace}".encode("utf-8"), digest_size=8
                ).digest()
                start = (int.from_bytes(digest, "little") + self._active_epoch) % len(items)
                return items[start:] + items[:start]

            remaining = (
                whole + rotate(non_chart, "non-chart")
                + rotate(chart, "chart") + other
            )

        def take(predicate):
            for index, sample in enumerate(remaining):
                if predicate(self._metadata(sample)):
                    return remaining.pop(index)
            return None

        def fill_bundle(prefer_whole=False):
            bundle = []
            if prefer_whole:
                whole = take(lambda meta: meta.get("view_type") == "whole")
                if whole is not None:
                    bundle.append(whole)
            if len(bundle) < self.max_samples:
                non_chart = take(lambda meta: meta.get("view_type") == "panel"
                                 and meta.get("content_type") != "chart")
                if non_chart is not None:
                    bundle.append(non_chart)
            if len(bundle) < self.max_samples:
                chart = take(lambda meta: meta.get("view_type") == "panel"
                             and meta.get("content_type") == "chart")
                if chart is not None:
                    bundle.append(chart)
            while remaining and len(bundle) < self.max_samples:
                bundle.append(remaining.pop(0))
            return bundle

        first = True
        while remaining:
            bundle = fill_bundle(prefer_whole=first)
            first = False
            if bundle:
                yield bundle

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        self._active_epoch = epoch
        rng = random.Random(pytorch_worker_seed(epoch) + self.seed)
        # Cache chunk types: reparsing every candidate's JSON dominates CPU.
        # A buffered chunk is immutable, so its signature is safe to reuse.
        # This preserves exactly the same weighted sampling distribution.
        buffer = []
        last_emitted_id = None
        recent_types = deque()
        recent_counts = Counter()

        def chunk_types(chunk):
            types = []
            for sample in chunk:
                meta = self._metadata(sample)
                if meta.get("view_type") == "panel":
                    types.append(str(meta.get("content_type") or "unknown"))
            return types or ["__whole_only__"]

        def remember(types):
            if self.balance_window <= 0:
                return
            for content_type in types:
                recent_types.append(content_type)
                recent_counts[content_type] += 1
                while len(recent_types) > self.balance_window:
                    expired = recent_types.popleft()
                    recent_counts[expired] -= 1
                    if recent_counts[expired] <= 0:
                        del recent_counts[expired]

        def emit_random():
            nonlocal last_emitted_id
            # In the steady state the buffer contains hundreds of chunks while
            # only balance_candidates of them are scored. Rejection sampling is
            # uniform over the same eligible indices without scanning the buffer.
            candidate_limit = self.balance_candidates
            if candidate_limit and len(buffer) > candidate_limit:
                candidates = []
                selected = set()
                attempts = 0
                max_attempts = len(buffer) * 4
                while len(candidates) < candidate_limit and attempts < max_attempts:
                    attempts += 1
                    index = rng.randrange(len(buffer))
                    if index in selected or buffer[index][0] == last_emitted_id:
                        continue
                    selected.add(index)
                    candidates.append(index)
                if len(candidates) < candidate_limit:
                    # Fall back to a full scan after bounded sampling.
                    candidates = [
                        index for index, (figure_id, _, _) in enumerate(buffer)
                        if figure_id != last_emitted_id
                    ] or list(range(len(buffer)))
            else:
                candidates = [
                    index for index, (figure_id, _, _) in enumerate(buffer)
                    if figure_id != last_emitted_id
                ]
                if not candidates:
                    candidates = list(range(len(buffer)))
            if self.balance_power > 0 and self.balance_window > 0:
                weights = []
                for candidate in candidates:
                    types = buffer[candidate][2]
                    weights.append(sum(
                        (recent_counts[content_type] + 1) ** (-self.balance_power)
                        for content_type in types
                    ) / len(types))
                index = rng.choices(candidates, weights=weights, k=1)[0]
            else:
                index = rng.choice(candidates)
            figure_id, chunk, types = buffer.pop(index)
            last_emitted_id = figure_id
            remember(types)
            for item in chunk:
                yield item

        current_id = None
        current_group = []
        for sample in src:
            figure_id = self._figure_id(sample)
            if current_group and figure_id != current_id:
                for chunk in self._chunks(current_group):
                    buffer.append((current_id, chunk, chunk_types(chunk)))
                    if len(buffer) >= self.bufsize and len({item[0] for item in buffer}) > 1:
                        yield from emit_random()
                current_group = []
            current_id = figure_id
            current_group.append(sample)

        for chunk in self._chunks(current_group):
            buffer.append((current_id, chunk, chunk_types(chunk)))
        while buffer:
            yield from emit_random()

class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


class ResumeFastForward(wds.PipelineStage):
    """Use cheap placeholders for samples discarded by mid-epoch resume."""

    marker = b"__bio_resume_fast_forward__"

    def __init__(self, skip_batches, batch_size, workers):
        self.skip_batches = int(skip_batches)
        self.batch_size = int(batch_size)
        self.workers = max(1, int(workers))

    def run(self, src):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else self.workers
        if not hasattr(self, "_remaining_batches"):
            base, remainder = divmod(self.skip_batches, worker_count)
            self._remaining_batches = base + int(worker_id < remainder)
        image_extensions = {"jpg", "jpeg", "png", "webp"}
        pending = []

        for sample in src:
            pending.append(sample)
            if len(pending) < self.batch_size:
                continue
            if self._remaining_batches > 0:
                for item in pending:
                    for key in list(item):
                        if key.lower() in image_extensions:
                            del item[key]
                    item["skip"] = self.marker
                    item["txt"] = self.marker
                    # Placeholders use the same metadata shape as real samples.
                    item["json"] = b'{"_bio":[0,0,0,0,0,0,0,0,0]}'
                self._remaining_batches -= 1
            yield from pending
            pending = []
        # Match wds.batched(..., partial=False): do not carry a partial batch
        # across a source restart, and only decrement for batches actually emitted.


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    use_bio_metadata = is_train and getattr(args, "bioloom_loss", False)
    excluded_articles = set()
    excluded_articles_path = getattr(args, "bio_excluded_articles", None)
    if use_bio_metadata and excluded_articles_path:
        with open(excluded_articles_path, encoding="utf-8") as handle:
            excluded_articles = {
                line.strip() for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            }

    def keep_uncontaminated_article(sample):
        metadata = FigureGroupShuffle._metadata(sample)
        return str(metadata.get("article_id") or "") not in excluded_articles

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=args.wds_shard_shuffle_size,
                    initial=args.wds_shard_shuffle_initial,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.append(
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow  # wds.tarfile_to_samples(handler=log_and_continue)
        )
        if excluded_articles:
            pipeline.append(wds.select(keep_uncontaminated_article))
        if use_bio_metadata and args.bio_group_shuffle_size > 0:
            pipeline.append(FigureGroupShuffle(
                bufsize=args.bio_group_shuffle_size,
                max_samples=args.bio_group_max_samples,
                seed=args.seed,
                epoch=shared_epoch,
                type_aware=args.bio_type_aware_groups,
                balance_power=args.bio_group_balance_power,
                balance_window=args.bio_group_balance_window,
                balance_candidates=args.bio_group_balance_candidates,
                rotate_panels=args.bio_rotate_panels,
            ))
        elif not use_bio_metadata:
            pipeline.append(wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ))
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    def stable_id(value):
        raw = str(value or "unknown").encode("utf-8")
        return int.from_bytes(
            hashlib.blake2b(raw, digest_size=8).digest(), "little"
        ) & ((1 << 63) - 1)

    embedded_bio = bool(getattr(args, "wds_embedded_bio", False))

    def encode_bio_metadata(meta):
        if isinstance(meta, bytes):
            meta = json.loads(meta.decode("utf-8"))
        elif isinstance(meta, str):
            meta = json.loads(meta)
        vector = meta.get("_bio") if embedded_bio else None
        if vector is not None:
            if not isinstance(vector, list) or len(vector) != 9:
                raise ValueError("expected a nine-field _bio vector")
            return torch.tensor(vector, dtype=torch.long)
        # Keep the legacy vector layout for shards that embed relation metadata.
        return torch.tensor([
            stable_id(meta.get("sample_id")),
            stable_id(meta.get("figure_id")),
            stable_id(meta.get("article_id")),
            stable_id(meta.get("content_type")),
            stable_id(meta.get("view_type")),
            stable_id("unknown"),
            stable_id("unknown"),
            0,
            10000,
        ], dtype=torch.long)

    sample_tuple = ("image", "text", "bio_metadata") if use_bio_metadata else ("image", "text")
    pretokenized = bool(getattr(args, "wds_pretokenized", False))
    # Some released shards carry the fixed-length token sidecar while older
    # shards contain only the source text. Prefer the stored tokens, but allow
    # the matching text field as a fallback so both serialize to the same
    # tokenizer output without dropping otherwise valid samples.
    text_member = "tok;txt" if pretokenized else "txt"
    pipeline.append(wds.select(filter_no_caption_or_no_image))
    resume_skip_batches = int(getattr(args, "bio_resume_fast_forward_batches", 0) or 0)
    if is_train and resume_skip_batches > 0:
        pipeline.append(ResumeFastForward(
            resume_skip_batches, args.batch_size, args.workers,
        ))

    def preprocess_or_skip(image):
        if isinstance(image, bytes) and image == ResumeFastForward.marker:
            if getattr(args, "wds_gpu_decode", False):
                return torch.frombuffer(bytearray(image), dtype=torch.uint8)
            return torch.empty((3, 1, 1), dtype=torch.float32)
        if getattr(args, "wds_gpu_decode", False):
            if not isinstance(image, bytes):
                raise TypeError("--wds-gpu-decode expects encoded image bytes")
            return torch.frombuffer(bytearray(image), dtype=torch.uint8)
        draft_size = int(getattr(args, "wds_pil_draft_size", 0) or 0)
        if draft_size > 0 and isinstance(image, bytes):
            # JPEG supports decoder-native 1/2, 1/4 and 1/8 DCT scaling. Decode
            # only the resolution needed by the final 224px antialiased crop;
            # PNG/WebP safely fall back to their normal decoder. Source bytes,
            # crop distribution, interpolation and model input size are unchanged.
            with Image.open(io.BytesIO(image)) as encoded:
                encoded.draft("RGB", (draft_size, draft_size))
                image = encoded.convert("RGB")
        if getattr(args, "wds_tensor_decode", False) and isinstance(image, bytes):
            image = decode_image_tensor(image)
        if getattr(args, "wds_preprocessed_images", False):
            # The fast WDS variant is resized once offline. Reuse the original
            # RGB conversion, float conversion and normalization while
            # omitting only the leading RandomResizedCrop.
            transforms = getattr(preprocess_img, "transforms", None)
            if not transforms or len(transforms) < 2:
                raise ValueError("--wds-preprocessed-images requires a composed train transform")
            for transform in transforms[1:]:
                image = transform(image)
            return image
        return preprocess_img(image)

    def tokenize_or_skip(text):
        if isinstance(text, bytes):
            is_marker = text == ResumeFastForward.marker
            if not is_marker:
                text = text.decode("utf-8")
        else:
            is_marker = text == ResumeFastForward.marker.decode("ascii")
        if is_marker:
            return torch.empty((1,), dtype=torch.long)
        return tokenizer(text)[0]

    def decode_text(text):
        return text.decode("utf-8") if isinstance(text, bytes) else text

    def decode_tokens(payload):
        if not isinstance(payload, bytes) or len(payload) != 1024:
            raise ValueError(f"invalid pretokenized payload length: {len(payload)}")
        return torch.frombuffer(bytearray(payload), dtype=torch.uint16).to(torch.int64)

    def decode_tokens_or_text(payload):
        if isinstance(payload, bytes) and len(payload) == 1024:
            return decode_tokens(payload)
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return tokenizer(payload)[0]

    def tokenize_prepared_batch(batch):
        images, texts, *rest = batch
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        else:
            texts = list(texts)
        tokens = tokenizer(texts)
        return (images, tokens, *rest)

    def gpu_decode_collation(samples):
        """Keep variable-length compressed JPEG tensors as a list.

        Metadata is still stacked normally. This prevents DataLoader workers
        from decoding and transferring a full float image batch through IPC.
        """
        columns = list(zip(*samples))
        text_values = list(columns[1])
        batched_text = torch.stack(text_values) if torch.is_tensor(text_values[0]) else text_values
        output = [list(columns[0]), batched_text]
        for column in columns[2:]:
            values = list(column)
            output.append(torch.stack(values) if torch.is_tensor(values[0]) else values)
        return tuple(output)

    rename_kwargs = dict(image="skip;jpg;png;jpeg;webp", text=text_member)
    map_kwargs = dict(image=preprocess_or_skip)
    if use_bio_metadata:
        rename_kwargs["bio_metadata"] = "json"
        map_kwargs["bio_metadata"] = encode_bio_metadata
    worker_batch_tokenize = bool(getattr(args, "wds_worker_batch_tokenize", False))
    if pretokenized:
        map_kwargs["text"] = decode_tokens_or_text
    else:
        map_kwargs["text"] = (
            decode_text
            if getattr(args, "wds_batch_tokenize", False) or worker_batch_tokenize
            else tokenize_or_skip
        )

    if (not getattr(args, "wds_tensor_decode", False)
            and not getattr(args, "wds_gpu_decode", False)
            and int(getattr(args, "wds_pil_draft_size", 0) or 0) <= 0):
        pipeline.append(wds.decode("pilrgb", handler=log_and_continue))
    batch_collation = gpu_decode_collation if getattr(args, "wds_gpu_decode", False) else wds.filters.default_collation_fn
    worker_batch_size = int(getattr(args, "wds_worker_batch_size", 0) or 0)
    if worker_batch_size <= 0:
        worker_batch_size = args.batch_size
    if worker_batch_size > args.batch_size or args.batch_size % worker_batch_size != 0:
        raise ValueError("--wds-worker-batch-size must divide --batch-size and be no larger than it")
    pipeline.extend([
        wds.rename(**rename_kwargs),
        wds.map_dict(**map_kwargs),
        wds.to_tuple(*sample_tuple),
        wds.batched(worker_batch_size, collation_fn=batch_collation, partial=not is_train)
    ])
    if worker_batch_tokenize and not pretokenized:
        pipeline.append(wds.map(tokenize_prepared_batch))
    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        microbatches_per_batch = args.batch_size // worker_batch_size
        num_worker_batches = round_fn(
            num_batches * microbatches_per_batch / num_workers
        )  # worker-side micro-batches per worker
        num_batches = num_worker_batches * num_workers // microbatches_per_batch
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.wds_prefetch_factor if args.workers > 0 else None,
        # The training loop uses non-blocking device copies. Pinning worker
        # output allows H2D transfer to overlap with compute without changing
        # decoded pixels or augmentation semantics.
        pin_memory=True,
    )
    if worker_batch_size != args.batch_size:
        # Combine worker chunks into full batches.
        dataloader = dataloader.unbatched().batched(
            args.batch_size,
            collation_fn=batch_collation,
            partial=not is_train,
        )
    if (getattr(args, "wds_batch_tokenize", False)
            and not worker_batch_tokenize and not pretokenized):
        # Run batch tokenization outside DataLoader worker processes.
        dataloader = dataloader.map(tokenize_prepared_batch)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
