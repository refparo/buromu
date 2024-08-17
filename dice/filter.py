from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Generator, Literal, cast

import numpy as np

from .expr import (
  BinaryCond,
  Coin,
  Combine,
  Compare,
  Cond,
  CountCond,
  DFace,
  DFate,
  DModFace,
  ExprFormatter,
  Filter,
  Not,
  Repeat,
  SimpleCond,
  Unique,
)
from .roll import (
  DArray,
  DieRoller,
  OrigDie,
  Rolled,
  RolledPool,
  rolled_die_to_str,
  signed_number_to_int,
)
from .token import Number, Span

type FlatPool = (
  FlatDice | Combine[FlatPool] | Filter[FlatPool] | Unique[FlatPool]
)


@dataclass(frozen=True)
class FlatDice:
  data: np.ndarray
  dice: list[OrigDie | int]
  span: Span


@dataclass(frozen=True)
class TooManyRerolls(Exception):
  span: Span


def flatten_pool(pool: RolledPool, shape: list[int]) -> FlatPool:
  match pool:
    case Number(value, span):
      shape.append(1)
      data = np.full(shape, value, dtype=np.int64)
      shape.pop()
      return FlatDice(data, [value], span)
    case Rolled(orig, data, span):
      return FlatDice(data, [orig], span)
    case Repeat(die, Number(count, _), span):
      if count == 0:
        shape.append(0)
        data = np.zeros(shape, dtype=np.int64)
        shape.pop()
        return FlatDice(data, [], span)
      else:
        match die:
          case Number(value, _):
            shape.append(1)
            data = np.full(shape, value, dtype=np.int64)
            shape.pop()
            return FlatDice(data, [value] * count, span)
          case Rolled(orig, data, _):
            return FlatDice(data, [orig] * count, span)
    case Combine(pools, span):
      flattened = [flatten_pool(pool, shape) for pool in pools]
      if all(isinstance(pool, FlatDice) for pool in flattened):
        shape.append(0)
        data = [np.zeros(shape, dtype=np.int64)]
        shape.pop()
        dice = []
        for each in flattened:
          each = cast(FlatDice, each)
          data.append(each.data)
          dice.extend(each.dice)
        return FlatDice(np.concat(data, axis=-1), dice, span)
      else:
        return Combine(flattened, span)
    case Filter(op, pool, cond, span):
      return Filter(op, flatten_pool(pool, shape), cond, span)
    case Unique(pool, span):
      return Unique(flatten_pool(pool, shape), span)


@dataclass
class SorterUnsorter:
  sorted: np.ndarray
  result: np.ndarray
  unsorted: np.ndarray


@contextmanager
def sort_unsort(
  arr: np.ndarray,
  axis: int = -1,
  kind: Literal["quicksort", "mergesort", "heapsort", "stable"] = "stable",
):
  sorted_idxs = np.argsort(arr, axis=axis, kind=kind)
  sorted = np.take_along_axis(arr, sorted_idxs, axis=axis)
  sorter = SorterUnsorter(sorted, sorted, arr)
  yield sorter
  if sorter.result is not sorted:
    unsort_idxs = np.argsort(sorted_idxs, axis=-1)
    sorter.unsorted = np.take_along_axis(sorter.result, unsort_idxs, axis=-1)


@lru_cache
def max_of_die(die: OrigDie | int) -> int:
  match die:
    case int(x):
      return x
    case DFace(Number(face, _), _):
      return face
    case DModFace(Number(face, _), _):
      return face - 1
    case DArray(faces_array, _):
      return int(np.max(faces_array))
    case DFate() | Coin():
      return 1


@lru_cache
def min_of_die(die: OrigDie | int) -> int:
  match die:
    case int(x):
      return x
    case DFace():
      return 1
    case DModFace() | Coin():
      return 0
    case DArray(faces_array, _):
      return int(np.min(faces_array))
    case DFate():
      return -1


def calculate_mask(
  data: np.ndarray, dice: list[int | OrigDie], cond: Cond
) -> np.ndarray:
  match cond:
    case Compare(comp, to, _):
      num = signed_number_to_int(to)
      match comp:
        case Compare.Comparator.EQ:
          return data == num
        case Compare.Comparator.GT:
          return data > num
        case Compare.Comparator.GE:
          return data >= num
        case Compare.Comparator.LT:
          return data < num
        case Compare.Comparator.LE:
          return data <= num
    case SimpleCond(kind, _):
      match kind:
        case SimpleCond.Kind.ODD:
          return data % 2 == 1
        case SimpleCond.Kind.EVEN:
          return data % 2 == 0
        case SimpleCond.Kind.MAX:
          min_values = np.array([max_of_die(die) for die in dice]).reshape(
            (1,) * (data.ndim - 1) + (len(dice),)
          )
          max_of_die.cache_clear()
          return data == min_values
        case SimpleCond.Kind.MIN:
          min_values = np.array([min_of_die(die) for die in dice]).reshape(
            (1,) * (data.ndim - 1) + (len(dice),)
          )
          min_of_die.cache_clear()
          return data == min_values
        case SimpleCond.Kind.UNIQ:
          if data.shape[-1] <= 1:
            return np.ones(data.shape, dtype=np.bool)
          else:
            with sort_unsort(data) as sorter:
              duplicates = sorter.sorted[..., 1:] == sorter.sorted[..., :-1]
              sorter.result = np.ma.concatenate(
                (np.zeros(data.shape[:-1] + (1,), dtype=np.bool), duplicates),
                axis=-1,
              )
              sorter.result[
                np.ma.concatenate(
                  (duplicates, np.zeros(data.shape[:-1] + (1,), dtype=np.bool)),
                  axis=-1,
                )
              ] = True
            return ~sorter.unsorted
        case SimpleCond.Kind.DUPMAX:
          if data.shape[-1] <= 1:
            return np.ones(data.shape, dtype=np.bool)
          else:
            with sort_unsort(data) as sorter:
              duplicates = sorter.sorted[..., 1:] == sorter.sorted[..., :-1]
              edges = np.diff(duplicates, axis=-1, prepend=0, append=0)
              edges = np.ma.masked_array(edges, np.ma.getmask(sorter.sorted))
              # reshape to make sure that ndim=2, will reshape back later
              edges = edges.reshape((-1, data.shape[-1]))
              rep_idxs, starts = np.nonzero(edges == 1)
              if len(rep_idxs) == 0:
                return np.ones(data.shape, dtype=np.bool)
              stops = np.nonzero(edges == -1)[1] + 1
              interval_lengths = stops - starts
              interval_counts = np.bincount(rep_idxs)
              interval_mask = (
                np.arange(interval_counts.max()) < interval_counts[:, None]
              )
              tmp = np.zeros(interval_mask.shape, dtype=np.int64)
              tmp[interval_mask] = interval_lengths
              mask = np.zeros(edges.shape, dtype=np.int8)
              select = interval_mask & (
                tmp == np.max(tmp, axis=1, keepdims=True)
              )
              tmp[interval_mask] = starts
              mask[np.nonzero(select)[0], tmp[select]] = 1
              tmp[interval_mask] = stops
              select &= tmp < data.shape[-1]
              mask[np.nonzero(select)[0], tmp[select]] += -1
              mask[np.argwhere(interval_counts == 0), 0] = 1
              mask[np.arange(len(interval_counts), mask.shape[0]), 0] = 1
              # finally reshape back
              sorter.result = (
                np.cumsum(mask, axis=-1).astype(np.bool).reshape(data.shape)
              )
            return sorter.unsorted
    case CountCond(kind, number, _):
      match number:
        case Number(count, _):
          pass
        case None:
          count = None
      match kind:
        case CountCond.Kind.DUP:
          if count is None:
            count = 2
          if count <= 1:
            return np.ones(data.shape, dtype=np.bool)
          elif data.shape[-1] < count:
            return np.zeros(data.shape, dtype=np.bool)
          else:
            with sort_unsort(data) as sorter:
              duplicates = sorter.sorted[..., 1:] == sorter.sorted[..., :-1]
              edges = np.diff(duplicates, axis=-1, prepend=0, append=0)
              edges = np.ma.masked_array(edges, np.ma.getmask(sorter.sorted))
              edges = edges.reshape((-1, data.shape[-1]))
              rep_idxs, starts = np.nonzero(edges == 1)
              if len(rep_idxs) == 0:
                return np.zeros(data.shape, dtype=np.bool)
              stops = np.nonzero(edges == -1)[1] + 1
              selected = (stops - starts) >= count
              mask = np.zeros(edges.shape, dtype=np.int8)
              mask[rep_idxs[selected], starts[selected]] = 1
              selected &= stops < data.shape[-1]
              mask[rep_idxs[selected], stops[selected]] += -1
              sorter.result = np.cumsum(mask, axis=-1).astype(np.bool)
            return sorter.unsorted
        case CountCond.Kind.HIGH:
          if count == 0 or data.shape[-1] == 0:
            return np.zeros(data.shape, dtype=np.bool)
          else:
            if count == 1 or count is None:
              idxs = np.argmax(data, axis=-1, keepdims=True)
            else:
              idxs = np.argsort(-data, axis=-1, kind="stable")[..., :count]
            mask = np.zeros(data.shape, dtype=np.bool)
            np.put_along_axis(mask, idxs, True, axis=-1)
            return mask
        case CountCond.Kind.LOW:
          if count == 0 or data.shape[-1] == 0:
            return np.zeros(data.shape, dtype=np.bool)
          else:
            if count == 1 or count is None:
              idxs = np.argmin(data, axis=-1, keepdims=True)
            else:
              idxs = np.argsort(data, axis=-1, kind="stable")[..., :count]
            mask = np.zeros(data.shape, dtype=np.bool)
            np.put_along_axis(mask, idxs, True, axis=-1)
            return mask
    case BinaryCond(op, lhs, rhs, _):
      match op:
        case BinaryCond.Operator.AND:
          return (
            calculate_mask(data, dice, lhs)
            & calculate_mask(data, dice, rhs)
          )  # fmt: skip
        case BinaryCond.Operator.OR:
          return (
            calculate_mask(data, dice, lhs)
            | calculate_mask(data, dice, rhs)
          )  # fmt: skip
        case BinaryCond.Operator.XOR:
          return (
            calculate_mask(data, dice, lhs)
            ^ calculate_mask(data, dice, rhs)
          )  # fmt: skip
    case Not(cond, _):
      return ~calculate_mask(data, dice, cond)


def drop_redundant(
  data: np.ndarray, dice: list[int | OrigDie]
) -> tuple[np.ndarray, list[int | OrigDie]]:
  mask = ~np.all(np.ma.getmask(data), axis=tuple(range(data.ndim - 1)))
  return data[..., mask], [die for i, die in enumerate(dice) if mask[i]]


def is_pool_relevant(cond: Cond):
  match cond:
    case (
      SimpleCond(SimpleCond.Kind.UNIQ | SimpleCond.Kind.DUPMAX, _)
      | CountCond()
    ):
      return True
    case BinaryCond(_, lhs, rhs, _):
      return is_pool_relevant(lhs) or is_pool_relevant(rhs)
    case Not(cond, _):
      return is_pool_relevant(cond)
    case _:
      return False


reroll_limit: Final = 50


class Filterer(DieRoller):
  def roll_where(
    self, mask: np.ndarray, dice: list[int | OrigDie]
  ) -> np.ndarray:
    # since we are only using this using rerolls, it is probably acceptable
    # to roll each die separately?
    arrays: list[np.ndarray] = []
    for die in dice:
      if isinstance(die, int):
        arrays.append(np.full(mask.shape[:-1], die, dtype=np.int64))
      else:
        match self.roll_die(die):
          case Number(value, _):
            arrays.append(np.full(mask.shape[:-1], value, dtype=np.int64))
          case Rolled(_, data, _):
            arrays.append(data)
    return np.ma.masked_array(np.stack(arrays, axis=-1), ~mask)

  def filter_filter(
    self, op: Filter.Operation, pool: FlatDice, cond: Cond, span: Span
  ) -> FlatDice:
    mask = calculate_mask(pool.data, pool.dice, cond)
    match op:
      case Filter.Operation.KEEP:
        return FlatDice(
          *drop_redundant(np.ma.masked_where(~mask, pool.data), pool.dice), span
        )
      case Filter.Operation.DROP:
        return FlatDice(
          *drop_redundant(np.ma.masked_where(mask, pool.data), pool.dice), span
        )
      case Filter.Operation.REROLL_ONCE:
        if np.any(mask):
          return FlatDice(
            np.where(mask, self.roll_where(mask, pool.dice), pool.data),
            pool.dice,
            span,
          )
        else:
          return FlatDice(pool.data, pool.dice, span)
      case Filter.Operation.REROLL:
        data = pool.data
        i = 0
        while np.any(mask):
          if i >= reroll_limit:
            raise TooManyRerolls(span)
          data = np.where(mask, self.roll_where(mask, pool.dice), pool.data)
          mask = calculate_mask(data, pool.dice, cond)
          i += 1
        return FlatDice(data, pool.dice, span)
      case Filter.Operation.EXPLODE:
        relevant: Final = is_pool_relevant(cond)
        axis: Final = tuple(range(pool.data.ndim - 1))
        acc_data = pool.data
        acc_dice = pool.dice
        dice = pool.dice
        prev_len = 0
        i = 0
        while np.any(mask):
          if i >= reroll_limit:
            raise TooManyRerolls(span)
          select = np.any(mask, axis=axis)
          dice = [die for i, die in enumerate(dice) if select[i]]
          rerolled = self.roll_where(mask[..., select], dice)
          prev_len = acc_data.shape[-1]
          acc_data = np.ma.concatenate((acc_data, rerolled), axis=-1)
          acc_dice.extend(dice)
          if relevant:
            mask = calculate_mask(acc_data, acc_dice, cond)[..., prev_len:]
          else:
            mask = calculate_mask(rerolled, dice, cond)
          i += 1
        return FlatDice(acc_data, acc_dice, span)
      case Filter.Operation.TEST:
        return FlatDice(
          np.ma.masked_array(
            mask.astype(np.int64), mask=np.ma.getmask(pool.data)
          ),
          [Coin(span)] * len(pool.dice),
          span,
        )

  def filter_unique(self, pool: FlatDice, span: Span) -> FlatDice:
    if pool.data.shape[-1] <= 1:
      return FlatDice(pool.data, pool.dice, span)
    else:
      with sort_unsort(pool.data) as sorter:
        sorter.result = np.ma.concatenate(
          (
            np.zeros(pool.data.shape[:-1] + (1,), dtype=np.bool),
            sorter.sorted[..., 1:] == sorter.sorted[..., :-1],
          ),
          axis=-1,
        )
      return FlatDice(
        *drop_redundant(
          np.ma.masked_where(sorter.unsorted, pool.data), pool.dice
        ),
        span,
      )

  def filter_step(self, pool: FlatPool) -> FlatPool:
    match pool:
      case FlatDice():
        return pool
      case Combine(pools, span):
        if all(isinstance(pool, FlatDice) for pool in pools):
          self.shape.append(0)
          data = [np.zeros(self.shape, dtype=np.int64)]
          self.shape.pop()
          dice = []
          for each in pools:
            each = cast(FlatDice, each)
            data.append(each.data)
            dice.extend(each.dice)
          return FlatDice(np.ma.concatenate(data, axis=-1), dice, span)
        else:
          return Combine([self.filter_step(pool) for pool in pools], span)
      case Filter(op, inner, cond, span):
        if isinstance(inner, FlatDice):
          return self.filter_filter(op, inner, cond, span)
        else:
          return Filter(op, self.filter_step(inner), cond, span)
      case Unique(inner, span):
        if isinstance(inner, FlatDice):
          return self.filter_unique(inner, span)
        else:
          return Unique(self.filter_step(inner), span)

  def filter_all(self, pool: FlatPool) -> FlatDice:
    match pool:
      case FlatDice():
        return pool
      case Combine(pools, span):
        self.shape.append(0)
        data = [np.zeros(self.shape, dtype=np.int64)]
        self.shape.pop()
        dice = []
        for each in pools:
          each = self.filter_all(each)
          data.append(each.data)
          dice.extend(each.dice)
        return FlatDice(np.concat(data, axis=-1), dice, span)
      case Filter(op, inner, cond, span):
        return self.filter_filter(op, self.filter_all(inner), cond, span)
      case Unique(inner, span):
        return self.filter_unique(self.filter_all(inner), span)


@dataclass
class FlattenedExprFormatter(ExprFormatter[FlatPool]):
  select: list[int]
  orig: str

  def pool_to_str(self, pool: FlatPool) -> Generator[str, None, None]:
    match pool:
      case FlatDice(data, _, _):
        if data.shape[-1] == 0:
          yield "{}"
        else:
          arr = data[*self.select, :]
          match np.ma.getmask(arr):
            case mask if mask is np.True_:
              arr = []
            case mask if mask is np.False_:
              pass
            case mask:
              arr = arr[~mask]
          if len(arr) == 0:
            yield "{}"
          elif len(arr) == 1:
            yield str(arr[0])
          else:
            yield "{"
            yield str(arr[0])
            for x in arr[1:]:
              yield ", "
              yield str(x)
            yield "}"
      case Combine(pools, _):
        yield "{"
        it = iter(pools)
        yield from self.pool_to_str(next(it))
        for pool in it:
          yield ", "
          yield from self.pool_to_str(pool)
        yield "}"
      case Filter(_, pool, _, (left, right)):
        yield self.orig[left : pool.span[0]]
        yield from self.pool_to_str(pool)
        yield self.orig[pool.span[1] : right]
      case Unique(pool, (left, right)):
        yield self.orig[left : pool.span[0]]
        yield from self.pool_to_str(pool)
        yield self.orig[pool.span[1] : right]
      case die:
        self.select.append(0)
        yield from rolled_die_to_str(die, self.select)
        self.select.pop()
