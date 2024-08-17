from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generator, Protocol

from .token import Number, Span

type Expr[P] = Binary[Expr[P]] | Neg[Expr[P]] | Pow[Expr[P]] | P
type Pool = (
  SignedNumber | Die | Repeat[Die] | Combine[Pool] | Filter[Pool] | Unique[Pool]
)
type Die = DFace | DModFace | DFaces | DFate | Coin
type Cond = Compare | SimpleCond | CountCond | BinaryCond | Not


@dataclass(frozen=True)
class Binary[E]:
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
  lhs: E
  rhs: E
  span: Span


@dataclass(frozen=True)
class Neg[E]:
  expr: E
  span: Span

@dataclass(frozen=True)
class Pow[E]:
  expr: E
  exp: Number
  span: Span

@dataclass(frozen=True)
class SignedNumber:
  sign: bool
  """True -> negative; False -> positive"""
  number: Number
  span: Span


@dataclass(frozen=True)
class Repeat[D]:
  die: D
  count: Number
  span: Span


@dataclass(frozen=True)
class Filter[P]:
  class Operation(Enum):
    KEEP = "k"
    DROP = "d"
    REROLL_ONCE = "r"
    REROLL = "rr"
    EXPLODE = "!"
    TEST = "?"

  op: Operation
  pool: P
  cond: Cond
  span: Span


@dataclass(frozen=True)
class Unique[P]:
  pool: P
  span: Span


@dataclass(frozen=True)
class Combine[P]:
  pools: list[P]
  span: Span


@dataclass(frozen=True)
class DFace:
  faces: Number
  span: Span


@dataclass(frozen=True)
class DModFace:
  faces: Number
  span: Span


@dataclass(frozen=True)
class DFaces:
  faces: list["Faces"]
  span: Span


@dataclass(frozen=True)
class Faces:
  start: SignedNumber
  step: SignedNumber | None
  stop: SignedNumber | None
  repeat: Number | None
  span: Span


@dataclass(frozen=True)
class DFate:
  span: Span


@dataclass(frozen=True)
class Coin:
  span: Span


@dataclass(frozen=True)
class Compare:
  class Comparator(Enum):
    EQ = "="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="

  comp: Comparator
  to: SignedNumber
  span: Span


@dataclass(frozen=True)
class SimpleCond:
  class Kind(Enum):
    ODD = "odd"
    EVEN = "even"
    MAX = "max"
    MIN = "min"
    UNIQ = "uniq"
    DUPMAX = "dupmax"

  kind: Kind
  span: Span


@dataclass(frozen=True)
class CountCond:
  class Kind(Enum):
    DUP = "dup"
    HIGH = "h"
    LOW = "l"

  kind: Kind
  count: Number | None
  span: Span


@dataclass(frozen=True)
class BinaryCond:
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
  span: Span


@dataclass(frozen=True)
class Not:
  cond: Cond
  span: Span


def map_expr[T, U](expr: Expr[T], f: Callable[[T], U]) -> Expr[U]:
  match expr:
    case Binary(op, lhs, rhs, span):
      return Binary(op, map_expr(lhs, f), map_expr(rhs, f), span)
    case Neg(expr, span):
      return Neg(map_expr(expr, f), span)
    case Pow(expr, exp, span):
      return Pow(map_expr(expr, f), exp, span)
    case pool:
      return f(pool)


class HasSpan(Protocol):
  @property
  def span(self) -> Span: ...


class ExprFormatter[P: HasSpan](ABC):
  select: list[int]
  orig: str

  def expr_to_str(self, expr: Expr[P]) -> Generator[str, None, None]:
    match expr:
      case Binary(_, lhs, rhs, (left, right)):
        yield self.orig[left : lhs.span[0]]
        yield from self.expr_to_str(lhs)
        yield self.orig[lhs.span[1] : rhs.span[0]]
        yield from self.expr_to_str(rhs)
        yield self.orig[rhs.span[1] : right]
      case Neg(expr, (left, right)):
        yield self.orig[left : expr.span[0]]
        yield from self.expr_to_str(expr)
        yield self.orig[expr.span[1] : right]
      case Pow(expr, _, (left, right)):
        yield self.orig[left : expr.span[0]]
        yield from self.expr_to_str(expr)
        yield self.orig[expr.span[1] : right]
      case pool:
        yield from self.pool_to_str(pool)

  @abstractmethod
  def pool_to_str(self, pool: P) -> Generator[str, None, None]: ...
