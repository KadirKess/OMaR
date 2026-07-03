from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class PrimitiveKind(StrEnum):
    NOTEHEAD = "notehead"
    STEM = "stem"
    BEAM = "beam"
    UNKNOWN = "unknown"


class SplitStatus(StrEnum):
    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return ``(x0, y0, x1, y1)`` with an exclusive lower-right corner."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


@dataclass(frozen=True)
class Primitive:
    kind: PrimitiveKind
    mask: np.ndarray
    confidence: float
    bbox: tuple[int, int, int, int] = field(init=False)

    def __post_init__(self):
        mask = np.asarray(self.mask, dtype=bool)
        bbox = mask_bbox(mask)
        if bbox is None:
            raise ValueError("a primitive mask cannot be empty")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "confidence", float(np.clip(self.confidence, 0, 1)))
        object.__setattr__(self, "bbox", bbox)


@dataclass(frozen=True)
class PrimitiveSplit:
    """Result of splitting one connected musical compound."""

    primitives: tuple[Primitive, ...]
    head_to_stem: tuple[tuple[int, int], ...] = ()
    stem_to_beam: tuple[tuple[int, int], ...] = ()
    status: SplitStatus = SplitStatus.REJECTED
    reasons: tuple[str, ...] = ()
    staff_space: float = 0.0
    staff_line: float = 0.0

    def of_kind(self, kind: PrimitiveKind) -> tuple[Primitive, ...]:
        return tuple(p for p in self.primitives if p.kind == kind)

    def class_mask(self, kind: PrimitiveKind) -> np.ndarray:
        if not self.primitives:
            raise ValueError("cannot infer mask shape from an empty result")
        result = np.zeros_like(self.primitives[0].mask, dtype=bool)
        for primitive in self.of_kind(kind):
            result |= primitive.mask
        return result
