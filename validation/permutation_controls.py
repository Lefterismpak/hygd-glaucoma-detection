"""Explicit block-label randomization for a separately designed future control.

This is NOT a retroactive replacement for any frozen HYGD experiment. It neither
fits a model nor computes a p value. Exchangeability is a design assumption that
must be justified for the actual data, not something this utility can prove.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def permute_label_blocks(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Exchange complete ordered label vectors within source/slot signatures.

    Required columns: source, group, slot, label. Group IDs must be global and
    confined to one source. Slots must be unique inside a group and identify
    comparable units (e.g. left/right eye). Blocks exchange only with another
    group with the same source and exact ordered slot signature. Unequal-size
    groups never exchange. The return value follows input ROW order.

    Conditioning on source and size/slots preserves those associations. The
    resulting null therefore does not test whether source or group size predicts
    disease, and does not prove biological independence. For a group-constant
    outcome with unequal repeats, a different group-level estimand/null may be
    appropriate; do not invent slot matches to make this function applicable.
    """
    required = {"source", "group", "slot", "label"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("A nonempty explicit block contract is required")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    normalized = frame[list(required)].copy().reset_index(drop=True)
    for column in ("source", "group", "slot"):
        if normalized[column].isna().any():
            raise ValueError("Source, group and slot must be nonempty")
        normalized[column] = normalized[column].astype(str)
        if normalized[column].str.strip().eq("").any():
            raise ValueError("Source, group and slot must be nonempty")
    labels = pd.to_numeric(normalized.label, errors="coerce").to_numpy(dtype=float)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be finite binary values")
    if normalized.duplicated(["group", "slot"]).any():
        raise ValueError("Slots must be unique within groups")
    if (normalized.groupby("group").source.nunique() > 1).any():
        raise ValueError("A group cannot cross sources")

    blocks = defaultdict(list)
    for _, group in normalized.groupby("group", sort=True):
        ordered = group.sort_values("slot", kind="stable")
        signature = (str(ordered.source.iloc[0]), tuple(ordered.slot))
        blocks[signature].append(ordered.index.to_numpy())
    if not any(len(bucket) >= 2 for bucket in blocks.values()):
        raise ValueError("No pair of blocks has an exchangeable source/slot signature")
    rng = np.random.default_rng(seed)
    shuffled = labels.astype(int).copy()
    for signature in sorted(blocks):
        bucket = blocks[signature]
        for destination, donor in zip(bucket, rng.permutation(len(bucket))):
            shuffled[destination] = labels[bucket[int(donor)]].astype(int)
    return shuffled
