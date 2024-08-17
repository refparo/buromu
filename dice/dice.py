from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

import numpy as np

np.seterr(all="raise")


class Dice(ABC):
  @abstractmethod
  def roll(self, repeat: int = 1) -> np.ndarray: ...
  @abstractmethod
  def mean_and_var(self) -> tuple[float, float]: ...
  @abstractmethod
  def min(self) -> int | None: ...
  @abstractmethod
  def max(self) -> int | None: ...
  @abstractmethod
  def q(self, p: np.ndarray) -> np.ndarray: ...
  @abstractmethod
  def hist(self) -> tuple[np.ndarray, np.ndarray]: ...


type AtomDie = Const | Die


@dataclass
class Const(Dice):
  value: Final[int]

  def roll(self, repeat: int = 1) -> np.ndarray:
    return np.full(repeat, self.value)

  def mean_and_var(self) -> tuple[float, float]:
    return self.value, 0

  def min(self) -> int:
    return self.value

  def max(self) -> int:
    return self.value

  def q(self, p: np.ndarray) -> np.ndarray:
    return np.full(len(p), self.value)

  def hist(self) -> tuple[np.ndarray, np.ndarray]:
    return np.array([self.value]), np.array([1])

  def __add__(self, other: AtomDie) -> AtomDie:
    match other:
      case Const(value):
        return Const(self.value + value)
      case Die(min_value, weights):
        return Die(self.value + min_value, weights)
      case _:
        return NotImplemented

  def __eq__(self, other: object) -> bool:
    if isinstance(other, Const):
      return self.value == other.value
    else:
      return NotImplemented

  def __hash__(self) -> int:
    return hash(self.value)


@dataclass
class Die(Dice):
  min_value: Final[int]
  weights: Final[np.ndarray]
  """ndarray of shape `(max - min + 1,)` and dtype `object`"""

  def __init__(self, min_value: int, weights: np.ndarray):
    assert weights.dtype == object, "weights must be of dtype object"
    assert len(weights.shape) == 1, "weights data must be 1-dimensional"
    assert np.all(weights >= 0), "weights must be non-negative"
    assert np.sum(weights) > 0, "die must contain at least 1 face"
    weights.setflags(write=False)
    self.min_value = min_value
    self.weights = weights

  def roll(self, repeat: int = 1) -> np.ndarray:
    return (
      np.random.choice(
        len(self.weights),
        size=repeat,
        p=(self.weights / np.sum(self.weights)).astype(float),
      )
      + self.min_value
    )

  def mean_and_var(self) -> tuple[float, float]:
    faces = np.arange(
      self.min_value, self.min_value + len(self.weights), dtype=int
    )
    avg = float(np.average(faces, weights=self.weights))
    return avg, float(np.average((faces - avg) ** 2, weights=self.weights))

  def min(self) -> int:
    return self.min_value

  def max(self) -> int:
    return self.min_value + len(self.weights) - 1

  def q(self, p: np.ndarray) -> np.ndarray:
    cumsum = np.cumsum(self.weights)
    search_targets = p * cumsum[-1]
    idxs_left = np.searchsorted(cumsum, search_targets, side="left")
    idxs_right = np.minimum(
      np.searchsorted(cumsum, search_targets, side="right"),
      len(cumsum) - 1,
    )
    return (idxs_left + idxs_right) / 2 + self.min_value

  def hist(self) -> tuple[np.ndarray, np.ndarray]:
    return (
      np.arange(self.min_value, self.min_value + len(self.weights), dtype=int),
      self.weights,
    )

  def __add__(self, other: AtomDie) -> AtomDie:
    match other:
      case Const(value):
        return Die(self.min_value + value, self.weights)
      case Die(min_value, weights):
        return Die(
          self.min_value + min_value,
          np.convolve(self.weights, weights),
        )
      case _:
        return NotImplemented

  def __eq__(self, other: object) -> bool:
    if isinstance(other, Die):
      return self is other or (
        self.min_value == other.min_value
        and len(self.weights) == len(other.weights)
        and bool(np.all(self.weights == other.weights))
      )
    else:
      return NotImplemented

  def __hash__(self) -> int:
    return hash((self.min_value, self.weights.data.tobytes()))


@dataclass
class Pool(Dice):
  components: Final[dict[AtomDie, int]]

  def __init__(self, components: dict[AtomDie, int]):
    assert all(
      count > 0 for count in components.values()
    ), "each component in a dice group must contain a positive number of dice"
    self.components = components

  def roll_components(
    self, repeat: int = 1
  ) -> list[tuple[AtomDie, np.ndarray]]:
    """
    :returns: an association list that maps each die to an ndarray of shape `(count, repeat)`
    """
    result: list[tuple[AtomDie, np.ndarray]] = []
    for die, count in self.components.items():
      result.append((die, die.roll(count * repeat).reshape(count, repeat)))
    return result

  def roll(self, repeat: int = 1) -> np.ndarray:
    return np.sum(
      [np.sum(arr, axis=0) for _, arr in self.roll_components(repeat)],
      axis=1,
    )

  def mean_and_var(self) -> tuple[float, float]:
    mean_var_count = [
      (die.mean_and_var(), count) for die, count in self.components.items()
    ]
    return (
      sum(mean * count for ((mean, _), count) in mean_var_count),
      sum(var * count for ((_, var), count) in mean_var_count),
    )

  def min(self) -> int | None:
    return sum(die.min() * count for die, count in self.components.items())

  def max(self) -> int | None:
    return sum(die.max() * count for die, count in self.components.items())

  def q(self, p: np.ndarray) -> np.ndarray:
    return self.flatten().q(p)

  def hist(self) -> tuple[np.ndarray, np.ndarray]:
    return self.flatten().hist()

  def complexity(self) -> int:
    cumprod = np.cumprod(
      [
        len(die.weights)
        for (die, count) in self.components.items()
        if isinstance(die, Die)
        for _ in range(count)
      ],
      dtype=object,
    )
    return int(np.sum(cumprod)) + sum(
      1 for die in self.components.keys() if isinstance(die, Const)
    )

  def flatten(self) -> AtomDie:
    const = 0
    die_exists = False
    die_acc = Die(0, np.array([1], dtype=object))
    for die, count in self.components.items():
      match die:
        case Const(value):
          const += value * count
        case Die():
          die_exists = True
          for _ in range(count):
            die_acc += die
    if die_exists:
      return die_acc + Const(const)
    else:
      return Const(const)
