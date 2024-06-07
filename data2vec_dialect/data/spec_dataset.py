# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import time
import io

import numpy as np
import torch
import torch.nn.functional as F


from fairseq.data import FairseqDataset
from fairseq.data.data_utils import compute_block_mask_1d, get_buckets, get_bucketed_sizes
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from typing import Optional
import kaldiio
import re

logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask=False,
        feature_encoder_spec: Optional[str] = None,
        mask_prob: float = 0.75,
        mask_prob_adjust: float = 0,
        mask_length: int = 1,
        inverse_mask: bool = False,
        require_same_masks: bool = True,
        clone_batch: int = 1,
        expand_adjacent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
        corpus_key=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

        self.is_compute_mask = compute_mask
        # list convolution layers' size: List[Tuple(in_c)]
        self.feature_encoder_spec = eval(feature_encoder_spec) if feature_encoder_spec is not None else None
        self._features_size_map = {}
        self.mask_prob = mask_prob
        self.mask_prob_adjust = mask_prob_adjust
        self.mask_length = mask_length
        self.inverse_mask = inverse_mask
        self.require_same_masks = require_same_masks
        self.clone_batch = clone_batch
        self.expand_adjacent = expand_adjacent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping
        self.corpus_key = corpus_key

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, t, target_size, dim=0):
        size = t.size(dim)
        diff = size - target_size
        if diff <= 0:
            return t

        start = np.random.randint(0, diff + 1)
        end = size - diff + start

        slices = []
        for d in range(dim):
            slices.append(slice(None))
        slices.append(slice(start, end))

        return t[slices]

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]
        B = len(sources)
        _, C = sources[0].shape

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(B, target_size, C)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape[:-1]).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff, C), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.corpus_key is not None:
            input["corpus_key"] = [self.corpus_key] * len(sources)
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if "precomputed_mask" in samples[0]:
            target_size = self._get_mask_indices_dims(target_size)
            collated_mask = torch.cat(
                [
                    self.crop_to_max_size(s["precomputed_mask"], target_size, dim=1)
                    for s in samples
                ],
                dim=0,
            )
            input["precomputed_mask"] = collated_mask

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        """
            compute sequence length after cnn subsampling
        """
        if size not in self.feature_encoder_spec:
            L_in = size
            for (_, kernel_size, stride) in self.feature_encoder_spec:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )

    def filter_indices_by_size(self, indices, max_sizes):
        return indices, []


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask=False,
        text_compression_level=TextCompressionLevel.none,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )

        self.text_compressor = TextCompressor(level=text_compression_level)

        skipped = 0
        self.fnames = []
        self.keys = []
        self.text_tokens = []
        text_tokens_len = []
        sizes = []
        self.skipped_indices = set()

        # load wenet format data
        # utt:key feat: feat_shape: text: token: tokenid: token_shape:
        REGEX_SPLIT = re.compile(r"\t")
        REGEX_FEAT = re.compile(r"feat:")
        with open(manifest_path, "r") as f:
            for i, line in enumerate(f):
                items = REGEX_SPLIT.split(line.strip())
                if len(items) != 7:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                # [T,C]
                shape = [int(x) for x in items[2].split(":")[1].split(",")]
                sz = shape[0]
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                key = items[0].split(":")[1]
                self.keys.append(self.text_compressor.compress(key))
                fname = REGEX_FEAT.sub("", items[1])
                self.fnames.append(self.text_compressor.compress(fname))
                sizes.append(sz)

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")
        logger.info(f"ATTENTION!!! skip indices: {self.skipped_indices}")

        self.sizes = np.array(sizes, dtype=np.int64)

        # utt-cmvn
        self.eps = 1e-5

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def postprocess(self, feats):
        assert feats.dim() == 2, feats.dim()

        # cmvn
        if self.normalize:
            m = feats.mean(dim=0)
            std = feats.std(dim=0)
            feats = (feats - m) / (std + self.eps)
        return feats

    @staticmethod
    def _bucket_3d_tensor(tensor, num_pad, value):
        """
            tensor: bxtxc, 3d-tensor, padding
        """
        return F.pad(tensor, (0, 0, 0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]
        B = len(sources)
        _, C = sources[0].shape

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(B, target_size, C)
        padding_mask = (
                torch.BoolTensor(collated_sources.shape[:-1]).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff, C), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.corpus_key is not None:
            input["corpus_key"] = [self.corpus_key] * len(sources)
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(1)
            if num_pad:
                input["source"] = self._bucket_3d_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if "precomputed_mask" in samples[0]:
            target_size = self._get_mask_indices_dims(target_size)
            collated_mask = torch.cat(
                [
                    self.crop_to_max_size(s["precomputed_mask"], target_size, dim=1)
                    for s in samples
                ],
                dim=0,
            )
            input["precomputed_mask"] = collated_mask

        out["net_input"] = input
        return out

    def __getitem__(self, index):
        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        path_or_fp = self.text_compressor.decompress(fn)
        retry = 3
        feats = None
        for i in range(retry):
            try:
                feats = kaldiio.load_mat(path_or_fp)
                break
            except Exception as e:
                logger.warning(
                    f"Failed to read {path_or_fp}: {e}. Sleeping for {1 * i}"
                )
                time.sleep(1 * i)

        if feats is None:
            raise Exception(f"Failed to load {path_or_fp}")

        feats = torch.from_numpy(feats).float()
        feats = self.postprocess(feats)

        v = {"id": index, "source": feats}

        if self.is_compute_mask:
            T = self._get_mask_indices_dims(feats.size(0))
            mask = compute_block_mask_1d(
                shape=(self.clone_batch, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
                expand_adjcent=self.expand_adjacent,
                mask_dropout=self.mask_dropout,
                non_overlapping=self.non_overlapping,
            )

            v["precomputed_mask"] = mask

        return v


class BinarizedAudioDataset(RawAudioDataset):
    def __init__(
        self,
        data_dir,
        split,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )

        from fairseq.data import data_utils, Dictionary

        self.fnames_dict = Dictionary.load(os.path.join(data_dir, "dict.txt"))

        root_path = os.path.join(data_dir, f"{split}.root")
        if os.path.exists(root_path):
            with open(root_path, "r") as f:
                self.root_dir = next(f).strip()
        else:
            self.root_dir = None

        fnames_path = os.path.join(data_dir, split)
        self.fnames = data_utils.load_indexed_dataset(fnames_path, self.fnames_dict)
        lengths_path = os.path.join(data_dir, f"{split}.lengths")

        with open(lengths_path, "r") as f:
            for line in f:
                sz = int(line.rstrip())
                assert (
                    sz >= min_sample_size
                ), f"Min sample size is not supported for binarized dataset, but found a sample with size {sz}"
                self.sizes.append(sz)

        self.sizes = np.array(self.sizes, dtype=np.int64)

        self.set_bucket_info(num_buckets)
        logger.info(f"loaded {len(self.fnames)} samples")

    def __getitem__(self, index):
        import soundfile as sf

        fname = self.fnames_dict.string(self.fnames[index], separator="")
        if self.root_dir:
            fname = os.path.join(self.root_dir, fname)

        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        v = {"id": index, "source": feats}

        if self.is_compute_mask:
            T = self._get_mask_indices_dims(feats.size(-1))
            mask = compute_block_mask_1d(
                shape=(self.clone_batch, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
                expand_adjcent=self.expand_adjacent,
                mask_dropout=self.mask_dropout,
                non_overlapping=self.non_overlapping,
            )

            v["precomputed_mask"] = mask

        return v
