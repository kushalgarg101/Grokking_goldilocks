"""Dataset builders and reproducibility utilities for grokking experiments."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset


def set_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch RNG seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TaskData:
    """Container for task-specific datasets and metadata."""

    train_ds: TensorDataset
    test_ds: TensorDataset
    vocab_size: int
    num_classes: int
    seq_len: int
    metadata: dict[str, Any]


def _split_tensor_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    train_fraction: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    """Create deterministic train/test splits from aligned tensors."""
    if len(x) != len(y):
        raise ValueError("x and y length mismatch")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(len(x), generator=gen)
    split_idx = int(len(x) * train_fraction)
    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]
    return (
        TensorDataset(x[train_idx], y[train_idx]),
        TensorDataset(x[test_idx], y[test_idx]),
    )


def build_modular_division_dataset(
    prime: int,
    train_fraction: float,
    seed: int,
) -> TaskData:
    """Build the modular division task: predict a / b (mod p)."""
    if prime < 3:
        raise ValueError("prime must be >= 3")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")

    slash_token = prime
    equals_token = prime + 1
    examples: list[tuple[int, int, int]] = []

    for a in range(prime):
        for b in range(1, prime):
            target = (a * pow(b, prime - 2, prime)) % prime
            examples.append((a, b, target))

    rng = random.Random(seed)
    rng.shuffle(examples)
    x = torch.tensor(
        [[a, slash_token, b, equals_token] for a, b, _ in examples],
        dtype=torch.long,
    )
    y = torch.tensor([target for _, _, target in examples], dtype=torch.long)
    train_ds, test_ds = _split_tensor_dataset(x, y, train_fraction=train_fraction, seed=seed)
    vocab_size = prime + 2
    return TaskData(
        train_ds=train_ds,
        test_ds=test_ds,
        vocab_size=vocab_size,
        num_classes=prime,
        seq_len=4,
        metadata={
            "task": "modular_division",
            "prime": prime,
            "special_tokens": {"slash": slash_token, "equals": equals_token},
        },
    )


def build_sparse_parity_dataset(
    bit_length: int,
    parity_k: int,
    dataset_size: int,
    train_fraction: float,
    seed: int,
) -> TaskData:
    """Build sparse parity task with a fixed latent subset of input bits."""
    if bit_length < 2:
        raise ValueError("bit_length must be >= 2")
    if not (1 <= parity_k <= bit_length):
        raise ValueError("parity_k must be in [1, bit_length]")
    if dataset_size <= 0:
        raise ValueError("dataset_size must be > 0")

    rng = np.random.default_rng(seed)
    subset = np.sort(rng.choice(bit_length, size=parity_k, replace=False))
    x_np = rng.integers(low=0, high=2, size=(dataset_size, bit_length), dtype=np.int64)
    y_np = np.sum(x_np[:, subset], axis=1) % 2

    x = torch.tensor(x_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.long)
    train_ds, test_ds = _split_tensor_dataset(x, y, train_fraction=train_fraction, seed=seed)
    return TaskData(
        train_ds=train_ds,
        test_ds=test_ds,
        vocab_size=2,
        num_classes=2,
        seq_len=bit_length,
        metadata={
            "task": "sparse_parity",
            "bit_length": bit_length,
            "parity_subset": subset.tolist(),
            "dataset_size": dataset_size,
        },
    )


def build_boolean_logic_dataset(
    input_bits: int,
    num_gates: int,
    dataset_size: int,
    train_fraction: float,
    seed: int,
) -> TaskData:
    """Build a random boolean-circuit evaluation dataset."""
    if input_bits < 2:
        raise ValueError("input_bits must be >= 2")
    if num_gates < 1:
        raise ValueError("num_gates must be >= 1")

    rng = random.Random(seed)
    ops = ("and", "or", "xor", "nand")
    pool = list(range(input_bits))
    gates: list[dict[str, Any]] = []
    for gate_idx in range(num_gates):
        if len(pool) >= 2:
            lhs, rhs = rng.sample(pool, 2)
        else:
            lhs = rhs = pool[0]
        op = rng.choice(ops)
        out_idx = input_bits + gate_idx
        gates.append({"op": op, "lhs": lhs, "rhs": rhs, "out": out_idx})
        pool.append(out_idx)

    total = 1 << input_bits
    if dataset_size <= 0 or dataset_size >= total:
        ids = np.arange(total, dtype=np.int64)
    else:
        ids = np.array(rng.sample(range(total), k=dataset_size), dtype=np.int64)

    bit_pos = np.arange(input_bits, dtype=np.int64)
    x_np = ((ids[:, None] >> bit_pos[None, :]) & 1).astype(np.int64)
    values = [x_np[:, i].copy() for i in range(input_bits)]
    for gate in gates:
        lhs = values[gate["lhs"]]
        rhs = values[gate["rhs"]]
        op = gate["op"]
        if op == "and":
            out = lhs & rhs
        elif op == "or":
            out = lhs | rhs
        elif op == "xor":
            out = lhs ^ rhs
        elif op == "nand":
            out = 1 - (lhs & rhs)
        else:
            raise ValueError(f"Unknown op: {op}")
        values.append(out.astype(np.int64))
    y_np = values[-1].astype(np.int64)

    x = torch.tensor(x_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.long)
    train_ds, test_ds = _split_tensor_dataset(x, y, train_fraction=train_fraction, seed=seed)
    return TaskData(
        train_ds=train_ds,
        test_ds=test_ds,
        vocab_size=2,
        num_classes=2,
        seq_len=input_bits,
        metadata={
            "task": "boolean_logic",
            "input_bits": input_bits,
            "num_gates": num_gates,
            "gates": gates,
            "dataset_size": int(len(ids)),
        },
    )


def build_task_data(args: argparse.Namespace) -> TaskData:
    """Dispatch dataset construction from parsed CLI arguments."""
    if args.task == "modular_division":
        return build_modular_division_dataset(
            prime=args.prime,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
    if args.task == "sparse_parity":
        return build_sparse_parity_dataset(
            bit_length=args.parity_bits,
            parity_k=args.parity_k,
            dataset_size=args.parity_dataset_size,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
    if args.task == "boolean_logic":
        return build_boolean_logic_dataset(
            input_bits=args.logic_input_bits,
            num_gates=args.logic_num_gates,
            dataset_size=args.logic_dataset_size,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
    raise ValueError(f"Unsupported task: {args.task}")

