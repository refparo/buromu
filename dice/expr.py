from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Final, Generator, Literal, cast, override

import numpy as np
from numpy import int64
from numpy.typing import NDArray

from .formattable import FormatContext, Formattable, TriviallyFormattable

# Evaluatable (see bottom for Expressions)


class Evaluatable(Formattable, ABC):
  @abstractmethod
  def eval(self, ctx: "RollContext", rec: bool) -> "Evaluatable": ...


@dataclass(frozen=True)
class Evaluated(Evaluatable):
  data: NDArray[int64]

  @override
  def eval(self, ctx: "RollContext", rec: bool) -> Evaluatable:
    return self

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    yield str(self.data[*ctx.select])


# Rollable


@dataclass
class RollContext:
  gen: np.random.Generator
  shape: list[int]


class Rollable(Evaluatable, ABC):
  @abstractmethod
  def roll(self, ctx: RollContext) -> "Rollable": ...
  @abstractmethod
  def roll_rec(self, ctx: RollContext) -> "RolledDice": ...

  @override
  def eval(self, ctx: RollContext, rec: bool) -> Evaluatable:
    return self.roll_rec(ctx) if rec else self.roll(ctx)


# RawRollable


class RawRollable(Rollable, ABC):
  @abstractmethod
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]: ...


@dataclass(frozen=True)
class TooManyDice(Exception):
  span: slice


dice_count_limit: Final = int64(1 << 17)


@dataclass(frozen=True)
class Repeat(RawRollable, TriviallyFormattable):
  n: int
  die: "SingleDie"
  span: slice

  @override
  def roll(self, ctx: RollContext) -> "RolledDice":
    return RolledDice(self.roll_raw(ctx), [self.die] * self.n)

  @override
  def roll_rec(self, ctx: RollContext) -> "RolledDice":
    return RolledDice(self.roll_raw(ctx), [self.die] * self.n)

  @override
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]:
    ctx.shape.append(self.n)
    if np.prod(ctx.shape) > dice_count_limit:
      raise TooManyDice(self.span)
    data = self.die.roll_raw(ctx)
    ctx.shape.pop()
    return data


# RolledDice


@dataclass(frozen=True)
class RolledDice(RawRollable, Evaluatable):
  data: NDArray[int64]
  components: list["SingleDie"]

  @override
  def roll(self, ctx: RollContext) -> "RolledDice":
    return RolledDice(self.roll_raw(ctx), self.components)

  @override
  def roll_rec(self, ctx: RollContext) -> "RolledDice":
    return RolledDice(self.roll_raw(ctx), self.components)

  @override
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]:
    data = np.stack([die.roll_raw(ctx) for die in self.components], axis=-1)
    if np.ma.is_masked(self.data):
      return np.ma.masked_array(data, np.ma.getmask(self.data))
    else:
      return data

  @override
  def eval(self, ctx: RollContext, rec: bool) -> Evaluatable:
    result = np.sum(self.data, axis=-1)
    return Evaluated(
      np.where(
        np.broadcast_to(np.ma.getmask(result), result.shape),
        0,
        result.astype(int64),
      )
    )

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    if self.data.shape[-1] == 0:
      yield "[]"
    else:
      data = self.data[*ctx.select, :]
      if np.ma.is_masked(self.data):
        data = data[np.broadcast_to(~np.ma.getmask(data), data.shape)]
      if len(data) == 0:
        yield "[]"
      elif len(data) == 1:
        yield str(data[0])
      else:
        yield "["
        it = iter(data)
        yield str(next(it))
        for x in it:
          too_long = yield ","
          if too_long:
            yield "…]"
            return
          yield str(x)
        yield "]"

  def optimize(self) -> "RolledDice":
    mask = ~np.all(
      np.ma.getmask(self.data), axis=tuple(range(self.data.ndim - 1))
    )
    return RolledDice(
      self.data[..., mask],
      [die for i, die in enumerate(self.components) if mask[i]],
    )


# SingleDie's


class SingleDie(RawRollable, TriviallyFormattable, ABC):
  @abstractmethod
  def max(self) -> int: ...
  @abstractmethod
  def min(self) -> int: ...

  @override
  def roll(self, ctx: RollContext) -> "RolledDice":
    ctx.shape.append(1)
    result = RolledDice(self.roll_raw(ctx), [self])
    ctx.shape.pop()
    return result

  @override
  def roll_rec(self, ctx: RollContext) -> "RolledDice":
    ctx.shape.append(1)
    result = RolledDice(self.roll_raw(ctx), [self])
    ctx.shape.pop()
    return result


@dataclass(frozen=True)
class Const(SingleDie):
  value: int
  span: slice

  @override
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]:
    return np.full(ctx.shape, self.value, dtype=int64)

  @override
  def max(self) -> int:
    return self.value

  @override
  def min(self) -> int:
    return self.value


@dataclass(frozen=True)
class DX(SingleDie):
  x: int
  span: slice

  @override
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]:
    return ctx.gen.choice(self.x, ctx.shape) + 1

  @override
  def max(self) -> int:
    return self.x

  @override
  def min(self) -> int:
    return 1


@dataclass(frozen=True)
class DModX(SingleDie):
  x: int
  span: slice

  @override
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]:
    return ctx.gen.choice(self.x, ctx.shape)

  @override
  def max(self) -> int:
    return self.x - 1

  @override
  def min(self) -> int:
    return 0


@dataclass
class DFaces(SingleDie):
  faces: NDArray[int64]
  span: slice
  _max: int | None = field(default=None, init=False, repr=False, compare=False)
  _min: int | None = field(default=None, init=False, repr=False, compare=False)

  @override
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]:
    return ctx.gen.choice(self.faces, ctx.shape)

  @override
  def max(self) -> int:
    if self._max is None:
      self._max = int(np.max(self.faces))
    return self._max

  @override
  def min(self) -> int:
    if self._min is None:
      self._min = int(np.min(self.faces))
    return self._min


@dataclass(frozen=True)
class DF(SingleDie):
  span: slice

  @override
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]:
    return ctx.gen.choice(3, ctx.shape) - 1

  @override
  def max(self) -> int:
    return 1

  @override
  def min(self) -> int:
    return -1


@dataclass(frozen=True)
class Coin(SingleDie):
  span: slice

  @override
  def roll_raw(self, ctx: RollContext) -> NDArray[int64]:
    return ctx.gen.choice(2, ctx.shape)

  @override
  def max(self) -> int:
    return 1

  @override
  def min(self) -> int:
    return 0


# Pool Operations


@dataclass(frozen=True)
class Combine(Rollable):
  components: list[Rollable]
  span: slice

  @override
  def roll(self, ctx: RollContext) -> Rollable:
    if all(isinstance(pool, RolledDice) for pool in self.components):
      ctx.shape.append(0)
      data: list[NDArray[int64]] = [np.zeros(ctx.shape, dtype=int64)]
      ctx.shape.pop()
      dice: list[SingleDie] = []
      for pool in self.components:
        pool = cast(RolledDice, pool)
        data.append(pool.data)
        dice.extend(pool.components)
      return RolledDice(np.ma.concatenate(data, axis=-1), dice)
    else:
      return Combine(
        [
          pool if isinstance(pool, RolledDice) else pool.roll(ctx)
          for pool in self.components
        ],
        self.span,
      )

  @override
  def roll_rec(self, ctx: RollContext) -> RolledDice:
    ctx.shape.append(0)
    data: list[NDArray[int64]] = [np.zeros(ctx.shape, dtype=int64)]
    ctx.shape.pop()
    dice: list[SingleDie] = []
    for pool in self.components:
      if not isinstance(pool, RolledDice):
        pool = pool.roll_rec(ctx)
      data.append(pool.data)
      dice.extend(pool.components)
    return RolledDice(np.ma.concatenate(data, axis=-1), dice)

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    if len(self.components) == 0:
      yield "[]"
    else:
      yield "["
      it = iter(self.components)
      yield from next(it).format(ctx)
      for pool in it:
        too_long = yield ","
        if too_long:
          yield "…]"
          return
        yield from pool.format(ctx)
      yield "]"


reroll_limit: Final = 50


@dataclass(frozen=True)
class TooManyRerolls(Exception):
  span: slice


@dataclass(frozen=True)
class Filter(Rollable):
  class Operation(Enum):
    KEEP = "k"
    DROP = "d"
    REROLL_ONCE = "r"
    REROLL = "rr"
    EXPLODE = "!"

  op: Operation
  inner: Rollable
  cond: "Cond | None"
  op_span: slice
  span: slice

  def filter(self, ctx: RollContext, dice: RolledDice) -> RolledDice:
    cond = self.cond or max_cond
    mask = cond.test(dice)
    match self.op:
      case Filter.Operation.KEEP:
        return RolledDice(
          np.ma.masked_where(~mask, dice.data), dice.components
        ).optimize()
      case Filter.Operation.DROP:
        return RolledDice(
          np.ma.masked_where(mask, dice.data), dice.components
        ).optimize()
      case Filter.Operation.REROLL_ONCE:
        if np.any(mask):
          return RolledDice(
            np.where(mask, dice.roll_raw(ctx), dice.data), dice.components
          )
        else:
          return dice
      case Filter.Operation.REROLL:
        i = 0
        while np.any(mask):
          if i >= reroll_limit:
            raise TooManyRerolls(self.span)
          dice = RolledDice(
            np.where(mask, dice.roll_raw(ctx), dice.data), dice.components
          )
          mask = cond.test(dice)
          i += 1
        return dice
      case Filter.Operation.EXPLODE:
        relevant = cond.is_pool_relevant()
        acc_dice = dice
        prev_len = 0
        i = 0
        while np.any(mask):
          if i >= reroll_limit:
            raise TooManyRerolls(self.span)
          dice = (
            RolledDice(np.ma.masked_where(~mask, dice.data), dice.components)
            .optimize()
            .roll(ctx)
          )
          prev_len = acc_dice.data.shape[-1]
          acc_dice = RolledDice(
            np.ma.concatenate([acc_dice.data, dice.data], axis=-1),
            acc_dice.components + dice.components,
          )
          if relevant:
            mask = cond.test(acc_dice)[..., prev_len:]
          else:
            mask = cond.test(dice)
          i += 1
        return acc_dice

  @override
  def roll(self, ctx: RollContext) -> Rollable:
    if isinstance(self.inner, RolledDice):
      return self.filter(ctx, self.inner)
    else:
      return Filter(
        self.op, self.inner.roll(ctx), self.cond, self.op_span, self.span
      )

  @override
  def roll_rec(self, ctx: RollContext) -> RolledDice:
    return self.filter(ctx, self.inner.roll_rec(ctx))

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    yield from self.inner.format(ctx)
    too_long = yield ctx.orig[self.op_span]
    if self.cond is not None:
      if too_long:
        yield "(…)"
      else:
        yield from self.cond.format(ctx)


@dataclass(frozen=True)
class Unique(Rollable):
  inner: Rollable
  op_span: slice
  # span: slice

  def unique(self, dice: RolledDice) -> RolledDice:
    if dice.data.shape[-1] <= 1:
      return dice
    else:
      with sorter_unsorter(dice.data) as sorter:
        sorter.result = np.ma.concatenate(
          (
            np.zeros(dice.data.shape[:-1] + (1,), dtype=np.bool),
            sorter.sorted[..., 1:] == sorter.sorted[..., :-1],
          ),
          axis=-1,
        )
      return RolledDice(
        np.ma.masked_where(sorter.unsorted, dice.data), dice.components
      ).optimize()

  @override
  def roll(self, ctx: RollContext) -> Rollable:
    if isinstance(self.inner, RolledDice):
      return self.unique(self.inner)
    else:
      return Unique(self.inner.roll(ctx), self.op_span)

  @override
  def roll_rec(self, ctx: RollContext) -> RolledDice:
    return self.unique(self.inner.roll_rec(ctx))

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    yield from self.inner.format(ctx)
    yield ctx.orig[self.op_span]


# Expressions


@dataclass(frozen=True)
class Count(Evaluatable):
  inner: Rollable
  cond: "Cond | None"
  op_span: slice
  # span: slice

  def count(self, dice: RolledDice) -> NDArray[int64]:
    mask = (self.cond or true_cond).test(dice)
    return np.sum(mask, axis=-1, dtype=int64)

  @override
  def eval(self, ctx: RollContext, rec: bool) -> Evaluatable:
    if isinstance(self.inner, RolledDice):
      return Evaluated(self.count(self.inner))
    else:
      return Count(
        self.inner.roll_rec(ctx) if rec else self.inner.roll(ctx),
        self.cond,
        self.op_span,
      )

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    yield from self.inner.format(ctx)
    too_long = yield ctx.orig[self.op_span]
    if self.cond is not None:
      if too_long:
        yield "(…)"
      else:
        yield from self.cond.format(ctx)


@dataclass(frozen=True)
class Binary(Evaluatable):
  class Operator(Enum):
    PLUS = "+"
    MINUS = "-"
    MULT = "*"
    DIV = "/"
    MOD = "%"

    def prec(self):
      match self:
        case self.MULT | self.DIV | self.MOD:
          return 2
        case self.PLUS | self.MINUS:
          return 1

  op: Operator
  lhs: Evaluatable
  rhs: Evaluatable

  @override
  def eval(self, ctx: RollContext, rec: bool) -> Evaluatable:
    lhs = self.lhs.eval(ctx, rec)
    rhs = self.rhs.eval(ctx, rec)
    if isinstance(self.lhs, Rollable) or isinstance(self.rhs, Rollable):
      return Binary(self.op, lhs, rhs)
    elif isinstance(lhs, Evaluated) and isinstance(rhs, Evaluated):
      match self.op:
        case Binary.Operator.PLUS:
          return Evaluated(lhs.data + rhs.data)
        case Binary.Operator.MINUS:
          return Evaluated(lhs.data - rhs.data)
        case Binary.Operator.MULT:
          return Evaluated(lhs.data * rhs.data)
        case Binary.Operator.DIV:
          return Evaluated(lhs.data // rhs.data)
        case Binary.Operator.MOD:
          return Evaluated(lhs.data % rhs.data)
    else:
      return Binary(self.op, lhs, rhs)

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    if isinstance(self.lhs, Binary) and self.lhs.op.prec() < self.op.prec():
      yield "("
      yield from self.lhs.format(ctx)
      yield ")"
    else:
      yield from self.lhs.format(ctx)
    yield " "
    yield self.op.value
    yield " "
    if isinstance(self.rhs, Binary) and self.rhs.op.prec() < self.op.prec():
      yield "("
      yield from self.rhs.format(ctx)
      yield ")"
    else:
      yield from self.rhs.format(ctx)


@dataclass(frozen=True)
class Neg(Evaluatable):
  inner: Evaluatable

  @override
  def eval(self, ctx: RollContext, rec: bool) -> Evaluatable:
    inner = self.inner.eval(ctx, rec)
    if isinstance(self.inner, Rollable):
      return Neg(inner)
    elif isinstance(inner, Evaluated):
      return Evaluated(-inner.data)
    else:
      return Neg(inner)

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    yield "-"
    if isinstance(self.inner, Binary):
      yield "("
      yield from self.inner.format(ctx)
      yield ")"
    else:
      yield from self.inner.format(ctx)


@dataclass(frozen=True)
class Pow(Evaluatable):
  lhs: Evaluatable
  rhs: Evaluatable

  @override
  def eval(self, ctx: RollContext, rec: bool) -> Evaluatable:
    lhs = self.lhs.eval(ctx, rec)
    rhs = self.rhs.eval(ctx, rec)
    if isinstance(self.lhs, Rollable) or isinstance(self.rhs, Rollable):
      return Pow(lhs, rhs)
    elif isinstance(lhs, Evaluated) and isinstance(rhs, Evaluated):
      return Evaluated(lhs.data**rhs.data)
    else:
      return Pow(lhs, rhs)

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    if isinstance(self.lhs, (Binary, Neg)):
      yield "("
      yield from self.lhs.format(ctx)
      yield ")"
    else:
      yield from self.lhs.format(ctx)
    yield "**"
    if isinstance(self.rhs, Binary):
      yield "("
      yield from self.rhs.format(ctx)
      yield ")"
    else:
      yield from self.rhs.format(ctx)


# Conditions


class sorter_unsorter(AbstractContextManager):
  _sorted_idxs: NDArray
  sorted: NDArray
  result: NDArray
  unsorted: NDArray

  def __init__(
    self,
    arr: NDArray[int64],
    axis: int = -1,
    kind: Literal["quicksort", "mergesort", "heapsort", "stable"] = "stable",
  ):
    self._sorted_idxs = np.argsort(arr, axis=axis, kind=kind)
    self.sorted = np.take_along_axis(arr, self._sorted_idxs, axis=axis)
    self.result = self.sorted
    self.unsorted = arr

  def __enter__(self) -> "sorter_unsorter":
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    traceback: TracebackType | None,
  ):
    unsort_idxs = np.argsort(self._sorted_idxs, axis=-1)
    self.unsorted = np.take_along_axis(self.result, unsort_idxs, axis=-1)


class Cond(TriviallyFormattable, ABC):
  @abstractmethod
  def test(self, dice: RolledDice) -> NDArray[np.bool]: ...

  def is_pool_relevant(self) -> bool:
    return False


@dataclass(frozen=True)
class Compare(Cond, TriviallyFormattable):
  class Comparator(Enum):
    EQ = "="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="

  comp: Comparator
  to: int
  span: slice

  @override
  def test(self, dice: RolledDice) -> NDArray[np.bool]:
    match self.comp:
      case Compare.Comparator.EQ:
        return dice.data == self.to
      case Compare.Comparator.GT:
        return dice.data > self.to
      case Compare.Comparator.GE:
        return dice.data >= self.to
      case Compare.Comparator.LT:
        return dice.data < self.to
      case Compare.Comparator.LE:
        return dice.data <= self.to


@dataclass(frozen=True)
class AtomCond(Cond, TriviallyFormattable):
  class Kind(Enum):
    ODD = "odd"
    EVEN = "even"
    MAX = "max"
    MIN = "min"
    UNIQ = "uniq"
    DUPMAX = "dupmax"
    TRUE = "true"

  kind: Kind
  span: slice

  @override
  def test(self, dice: RolledDice) -> NDArray[np.bool]:
    match self.kind:
      case AtomCond.Kind.ODD:
        return dice.data % 2 == 1
      case AtomCond.Kind.EVEN:
        return dice.data % 2 == 0
      case AtomCond.Kind.MAX:
        min_values = np.array([die.max() for die in dice.components]).reshape(
          (1,) * (dice.data.ndim - 1) + (len(dice.components),)
        )
        return dice.data == min_values
      case AtomCond.Kind.MIN:
        min_values = np.array([die.min() for die in dice.components]).reshape(
          (1,) * (dice.data.ndim - 1) + (len(dice.components),)
        )
        return dice.data == min_values
      case AtomCond.Kind.UNIQ:
        if dice.data.shape[-1] <= 1:
          return np.ones(dice.data.shape, dtype=np.bool)
        else:
          with sorter_unsorter(dice.data) as sorter:
            duplicates = sorter.sorted[..., 1:] == sorter.sorted[..., :-1]
            sorter.result = np.ma.concatenate(
              (
                np.zeros(dice.data.shape[:-1] + (1,), dtype=np.bool),
                duplicates,
              ),
              axis=-1,
            )
            sorter.result[
              np.ma.concatenate(
                (
                  duplicates,
                  np.zeros(dice.data.shape[:-1] + (1,), dtype=np.bool),
                ),
                axis=-1,
              )
            ] = True
          return ~sorter.unsorted
      case AtomCond.Kind.DUPMAX:
        if dice.data.shape[-1] <= 1:
          return np.ones(dice.data.shape, dtype=np.bool)
        else:
          with sorter_unsorter(dice.data) as sorter:
            duplicates = sorter.sorted[..., 1:] == sorter.sorted[..., :-1]
            edges = np.diff(duplicates, axis=-1, prepend=0, append=0)
            edges = np.ma.masked_array(edges, np.ma.getmask(sorter.sorted))
            # reshape to make sure that ndim=2, will reshape back later
            edges = edges.reshape((-1, dice.data.shape[-1]))
            rep_idxs, starts = np.nonzero(edges == 1)
            if len(rep_idxs) == 0:
              return np.ones(dice.data.shape, dtype=np.bool)
            stops = np.nonzero(edges == -1)[1] + 1
            interval_lengths = stops - starts
            interval_counts = np.bincount(rep_idxs)
            interval_mask = (
              np.arange(interval_counts.max()) < interval_counts[:, None]
            )
            tmp = np.zeros(interval_mask.shape, dtype=np.int64)
            tmp[interval_mask] = interval_lengths
            mask = np.zeros(edges.shape, dtype=np.int8)
            select = interval_mask & (tmp == np.max(tmp, axis=1, keepdims=True))
            tmp[interval_mask] = starts
            mask[np.nonzero(select)[0], tmp[select]] = 1
            tmp[interval_mask] = stops
            select &= tmp < dice.data.shape[-1]
            mask[np.nonzero(select)[0], tmp[select]] += -1
            mask[np.argwhere(interval_counts == 0), 0] = 1
            mask[np.arange(len(interval_counts), mask.shape[0]), 0] = 1
            # finally reshape back
            sorter.result = (
              np.cumsum(mask, axis=-1).astype(np.bool).reshape(dice.data.shape)
            )
          return sorter.unsorted
      case AtomCond.Kind.TRUE:
        return np.ones(dice.data.shape, dtype=np.bool)

  @override
  def is_pool_relevant(self) -> bool:
    return self.kind == AtomCond.Kind.UNIQ or self.kind == AtomCond.Kind.DUPMAX


@dataclass(frozen=True)
class CountingCond(Cond, TriviallyFormattable):
  class Kind(Enum):
    DUP = "dup"
    HIGH = "h"
    LOW = "l"

  kind: Kind
  count: int | None
  span: slice

  @override
  def test(self, dice: RolledDice) -> NDArray[np.bool]:
    count = self.count
    match self.kind:
      case CountingCond.Kind.DUP:
        if count is None:
          count = 2
        if count <= 1:
          return np.ones(dice.data.shape, dtype=np.bool)
        elif dice.data.shape[-1] < count:
          return np.zeros(dice.data.shape, dtype=np.bool)
        else:
          with sorter_unsorter(dice.data) as sorter:
            duplicates = sorter.sorted[..., 1:] == sorter.sorted[..., :-1]
            edges = np.diff(duplicates, axis=-1, prepend=0, append=0)
            edges = np.ma.masked_array(edges, np.ma.getmask(sorter.sorted))
            edges = edges.reshape((-1, dice.data.shape[-1]))
            rep_idxs, starts = np.nonzero(edges == 1)
            if len(rep_idxs) == 0:
              return np.zeros(dice.data.shape, dtype=np.bool)
            stops = np.nonzero(edges == -1)[1] + 1
            selected = (stops - starts) >= count
            mask = np.zeros(edges.shape, dtype=np.int8)
            mask[rep_idxs[selected], starts[selected]] = 1
            selected &= stops < dice.data.shape[-1]
            mask[rep_idxs[selected], stops[selected]] += -1
            sorter.result = np.cumsum(mask, axis=-1).astype(np.bool)
          return sorter.unsorted
      case CountingCond.Kind.HIGH:
        if count == 0 or dice.data.shape[-1] == 0:
          return np.zeros(dice.data.shape, dtype=np.bool)
        else:
          if count == 1 or count is None:
            idxs = np.argmax(dice.data, axis=-1, keepdims=True)
          else:
            idxs = np.argsort(-dice.data, axis=-1, kind="stable")[..., :count]
          mask = np.zeros(dice.data.shape, dtype=np.bool)
          np.put_along_axis(mask, idxs, True, axis=-1)
          return mask
      case CountingCond.Kind.LOW:
        if count == 0 or dice.data.shape[-1] == 0:
          return np.zeros(dice.data.shape, dtype=np.bool)
        else:
          if count == 1 or count is None:
            idxs = np.argmin(dice.data, axis=-1, keepdims=True)
          else:
            idxs = np.argsort(dice.data, axis=-1, kind="stable")[..., :count]
          mask = np.zeros(dice.data.shape, dtype=np.bool)
          np.put_along_axis(mask, idxs, True, axis=-1)
          return mask

  @override
  def is_pool_relevant(self) -> bool:
    return True


@dataclass(frozen=True)
class BinaryCond(Cond, TriviallyFormattable):
  class Operator(Enum):
    AND = "&"
    OR = "|"
    XOR = "^"

    def prec(self):
      match self:
        case self.AND:
          return 1
        case self.OR | self.XOR:
          return 0

  op: Operator
  lhs: Cond
  rhs: Cond
  span: slice

  @override
  def test(self, dice: RolledDice) -> NDArray[np.bool]:
    match self.op:
      case BinaryCond.Operator.AND:
        return self.lhs.test(dice) & self.rhs.test(dice)
      case BinaryCond.Operator.OR:
        return self.lhs.test(dice) | self.rhs.test(dice)
      case BinaryCond.Operator.XOR:
        return self.lhs.test(dice) ^ self.rhs.test(dice)

  @override
  def is_pool_relevant(self) -> bool:
    return self.lhs.is_pool_relevant() or self.lhs.is_pool_relevant()


@dataclass(frozen=True)
class Not(Cond, TriviallyFormattable):
  inner: Cond
  span: slice

  @override
  def test(self, dice: RolledDice) -> NDArray[np.bool]:
    return ~self.inner.test(dice)

  @override
  def is_pool_relevant(self) -> bool:
    return self.inner.is_pool_relevant()


max_cond: Final = AtomCond(AtomCond.Kind.MAX, slice(0))
true_cond: Final = AtomCond(AtomCond.Kind.TRUE, slice(0))
