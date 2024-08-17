from dataclasses import dataclass
from typing import Final, Generator

import numpy as np

from .expr import (
  Coin,
  Combine,
  DFace,
  DFaces,
  DFate,
  Die,
  DModFace,
  ExprFormatter,
  Faces,
  Filter,
  Pool,
  Repeat,
  SignedNumber,
  Unique,
)
from .token import Number, Span

type RolledPool = (
  RolledDie
  | Repeat[RolledDie]
  | Combine[RolledPool]
  | Filter[RolledPool]
  | Unique[RolledPool]
)
type RolledDie = Number | Rolled

type OrigDie = DFace | DModFace | DArray | DFate | Coin


@dataclass(frozen=True)
class Rolled:
  orig: OrigDie
  data: np.ndarray
  """ndarray of dtype `np.int64`"""
  span: Span


@dataclass(frozen=True)
class DArray:
  faces_array: np.ndarray
  span: Span


@dataclass(frozen=True)
class ZeroFacedDie(Exception):
  die: Die


@dataclass(frozen=True)
class InfiniteFaces(Exception):
  faces: Faces


@dataclass(frozen=True)
class TooManyFaces(Exception):
  die: DFaces


@dataclass(frozen=True)
class TooManyDice(Exception):
  pass


limit_dice: Final = np.int64(1 << 17)


@dataclass
class DieRoller:
  gen: np.random.Generator
  shape: list[int]

  def roll_die(self, die: Die | OrigDie) -> RolledDie:
    if np.prod(self.shape) > limit_dice:
      raise TooManyDice
    match die:
      case DFace(Number(value, _), span):
        if value == 0:
          raise ZeroFacedDie(die)
        elif value == 1:
          return Number(1, span)
        else:
          return Rolled(die, self.gen.choice(value, self.shape) + 1, span)
      case DModFace(Number(value, _), span):
        if value == 0:
          return Number(0, span)
        else:
          return Rolled(die, self.gen.choice(value, self.shape), span)
      case DFaces(_, span) as faces:
        faces_list = list(faces_to_arrays(faces))
        if len(faces_list) == 0:
          raise ZeroFacedDie(faces)
        faces_array = np.hstack(faces_list)
        if not np.diff(faces_array).any():  # if all face is the same
          return Number(faces_array[0], span)
        else:
          return Rolled(
            DArray(faces_array, span),
            self.gen.choice(faces_array, self.shape),
            span,
          )
      case DArray(faces_array, span):
        return Rolled(die, self.gen.choice(faces_array, self.shape), span)
      case DFate(span):
        return Rolled(die, self.gen.choice(3, self.shape) - 1, span)
      case Coin(span):
        return Rolled(die, self.gen.choice(2, self.shape), span)


@dataclass
class PoolRoller(DieRoller):
  gen: np.random.Generator
  shape: list[int]

  def roll_pool(self, pool: Pool) -> RolledPool:
    match pool:
      case SignedNumber(_, _, span) as number:
        return Number(signed_number_to_int(number), span)
      case Repeat(die, count, span):
        if count.value == 0:
          return Repeat(Number(0, die.span), count, span)
        else:
          self.shape.append(count.value)
          rolled = self.roll_die(die)
          self.shape.pop()
          return Repeat(rolled, count, span)
      case Combine(pools, span):
        return Combine([self.roll_pool(pool) for pool in pools], span)
      case Filter(op, pool, cond, span):
        return Filter(op, self.roll_pool(pool), cond, span)
      case Unique(pool, span):
        return Unique(self.roll_pool(pool), span)
      case die:
        self.shape.append(1)
        rolled = self.roll_die(die)
        self.shape.pop()
        return rolled


def signed_number_to_int(number: SignedNumber) -> int:
  return -number.number.value if number.sign else number.number.value


faces_limit: Final[np.int64] = np.int64(1 << 17)


def faces_to_arrays(faces: DFaces):
  face_count = 0
  for face in faces.faces:
    if face.repeat is None:
      repeat = 1
    else:
      repeat = face.repeat.value
      if repeat == 0:
        continue
    match faces_to_range(face):
      case (start, stop, step):
        face_count += (stop - start + step - 1) // step * repeat
        if face_count > faces_limit:
          raise TooManyFaces(faces)
        for _ in range(repeat):
          yield np.arange(start, stop, step, dtype=np.int64)
      case value:
        face_count += repeat
        if face_count > faces_limit:
          raise TooManyFaces(faces)
        yield np.full(repeat, value, dtype=np.int64)


def faces_to_range(faces: Faces):
  """:raises: IllegalFaces"""
  start = signed_number_to_int(faces.start)
  if faces.stop is None:
    return start
  else:
    stop = signed_number_to_int(faces.stop)
    if stop == start:
      return start
    else:
      sign = int(np.sign(stop - start))
      stop += sign
      if faces.step is None:
        step = sign
      else:
        step = signed_number_to_int(faces.step) - start
        if step * sign <= 0:
          raise InfiniteFaces(faces)
      return (start, stop, step)


@dataclass
class RolledExprFormatter(ExprFormatter[RolledPool]):
  select: list[int]
  orig: str

  def pool_to_str(self, pool: RolledPool) -> Generator[str, None, None]:
    match pool:
      case SignedNumber(_, _, span):
        yield self.orig[span[0] : span[1]]
      case Repeat(die, Number(repeat, _), _):
        yield "{"
        if repeat > 0:
          self.select.append(0)
          yield from rolled_die_to_str(die, self.select)
          for i in range(1, repeat):
            self.select[-1] = i
            yield ", "
            yield from rolled_die_to_str(die, self.select)
          self.select.pop()
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


def rolled_die_to_str(
  die: RolledDie, select: list[int]
) -> Generator[str, None, None]:
  match die:
    case Number(value, _):
      yield str(value)
    case Rolled(_, data, _):
      yield str(data[*select])
