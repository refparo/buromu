from dataclasses import dataclass, replace
from typing import Any, Final, TypeAliasType, cast

import numpy as np

from utils.peekable import Peekable

from .expr import (
  DF,
  DX,
  AtomCond,
  Binary,
  BinaryCond,
  Coin,
  Combine,
  Compare,
  Cond,
  Const,
  Count,
  CountingCond,
  DFaces,
  DModX,
  Evaluatable,
  Filter,
  Neg,
  Not,
  Pow,
  Repeat,
  Rollable,
  SingleDie,
  Unique,
)
from .token import Number, Simple, Token


@dataclass(frozen=True)
class ExpectError(Exception):
  type Expectation = TypeAliasType | type | Simple.Kind
  expected: Expectation | tuple[Expectation, ...]
  got: Token | None


# if failed to parse any prefix, all parsers must not consume anything,
# but raise an ExpectError expecting the whole subexpression (not prefixes)


def parse_expr(lex: Peekable[Token], prev_prec: int = 0) -> Evaluatable:
  """
  Expr -> PlusExpr
  PlusExpr ->
    | PlusExpr + PlusExpr
    | PlusExpr - PlusExpr
    | MultExpr
  MultExpr ->
    | MultExpr * MultExpr
    | MultExpr / MultExpr
    | MultExpr % MultExpr
    | UnaryExpr
  """
  lhs = parse_unary_expr(lex)
  while isinstance(tok := lex.peek(None), Simple):
    try:
      op = Binary.Operator[tok.kind.name]
      prec = op.prec()
      if prec <= prev_prec:
        break
    except KeyError:
      break
    next(lex)
    rhs = parse_expr(lex, prec)
    lhs = Binary(op, lhs, rhs)
  return lhs


def parse_unary_expr(lex: Peekable[Token]) -> Evaluatable:
  """
  UnaryExpr ->
    | - UnaryExpr
    | AtomExpr
  """
  match lex.peek(None):
    case Simple(Simple.Kind.MINUS, _):
      next(lex)
      expr = parse_atom_expr(lex)
      return Neg(expr)
    case _:
      return parse_atom_expr(lex)


def parse_atom_expr(lex: Peekable[Token]) -> Evaluatable:
  """
  AtomExpr ->
    | ( Expr )
    | Pool (# Cond?)?
    | AtomExpr ** UnaryExpr
  """
  match lex.peek(None):
    case Simple(Simple.Kind.LPAR, _):
      next(lex)
      expr = parse_expr(lex)
      match next(lex, None):
        case Simple(Simple.Kind.RPAR, _):
          lhs = expr
        case tok:
          if tok is not None:
            lex.put_back(tok)
          raise ExpectError(Simple.Kind.RPAR, tok)
    case tok:
      try:
        lhs = parse_pool(lex)
      except ExpectError as ex:
        if ex.expected == Rollable:
          raise ExpectError(Evaluatable, tok)
        else:
          raise
      match lex.peek(None):
        case Simple(Simple.Kind.COUNT, span):
          next(lex)
          try:
            cond = parse_cond(lex)
          except ExpectError as ex:
            if ex.expected == Cond:
              cond = None
            else:
              raise
          lhs = Count(lhs, cond, span)
  match lex.peek(None):
    case Simple(Simple.Kind.POW, _):
      next(lex)
      rhs = parse_unary_expr(lex)
      return Pow(lhs, rhs)
    case _:
      return lhs


def parse_pool(lex: Peekable[Token]) -> Rollable:
  """
  Pool ->
    | Pool k Cond
    | Pool d Cond
    | Pool r Cond
    | Pool rr Cond
    | Pool ! Cond?
    | Pool uniq
    | AtomPool
  """
  pool = parse_atom_pool(lex)
  while True:
    match lex.peek(None):
      case Simple(Simple.Kind.K, span):
        op = Filter.Operation.KEEP
      case Simple(Simple.Kind.D, span):
        op = Filter.Operation.DROP
      case Simple(Simple.Kind.R, span):
        op = Filter.Operation.REROLL_ONCE
      case Simple(Simple.Kind.RR, span):
        op = Filter.Operation.REROLL
      case Simple(Simple.Kind.BANG, span):
        next(lex)
        op = Filter.Operation.EXPLODE
        try:
          cond = parse_cond(lex)
        except ExpectError as ex:
          if ex.expected == Cond:
            cond = None
          else:
            raise
        pool = Filter(
          op,
          pool,
          cond,
          span,
          slice(pool.span.start, span.stop if cond is None else cond.span.stop),
        )
        continue
      case Simple(Simple.Kind.UNIQ, span):
        next(lex)
        pool = Unique(pool, span)
        continue
      case _:
        return pool
    next(lex)
    cond = parse_cond(lex)
    pool = Filter(op, pool, cond, span, slice(pool.span.start, cond.span.stop))


def parse_atom_pool(lex: Peekable[Token]) -> Rollable:
  """
  AtomPool ->
    | Die
    | SignedNumber
    | Number Die
    | [ Pool (, Pool)* ]
  """
  match lex.peek(None):
    case Simple(Simple.Kind.PLUS | Simple.Kind.MINUS, _) as tok:
      return parse_signed_number(lex)
    case Number(value, span) as number:
      next(lex)
      try:
        die = parse_die(lex)
        return Repeat(number.value, die, slice(span.start, die.span.stop))
      except ExpectError as ex:
        if ex.expected == SingleDie:
          return Const(value, span)
        else:
          raise
    case Simple(Simple.Kind.LBRACKET, left):
      next(lex)
      pools = [parse_pool(lex)]
      while True:
        match next(lex, None):
          case Simple(Simple.Kind.COMMA, _):
            pools.append(parse_pool(lex))
          case Simple(Simple.Kind.RBRACKET, right):
            return Combine(pools, slice(left.start, right.stop))
          case tok:
            if tok is not None:
              lex.put_back(tok)
            raise ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACKET), tok)
    case tok:
      try:
        return parse_die(lex)
      except ExpectError as ex:
        if ex.expected == SingleDie:
          raise ExpectError(Rollable, tok)
        else:
          raise


@dataclass(frozen=True)
class Faces:
  start: int
  stop: int | None  # stop = None means that this is a single face
  step: int
  repeat: int
  span: slice

  def count(self):
    if self.stop is None:
      return self.repeat
    else:
      return (self.stop - self.start + self.step - 1) // self.step * self.repeat

  def to_array(self):
    if self.stop is None:
      return np.full(self.repeat, self.start, dtype=np.int64)
    else:
      return np.tile(
        np.arange(self.start, self.stop, self.step, dtype=np.int64),
        self.repeat,
      )


@dataclass(frozen=True)
class ZeroFacedDie(Exception):
  span: slice


@dataclass(frozen=True)
class TooManyFaces(Exception):
  span: slice


faces_limit: Final[np.int64] = np.int64(1 << 17)


def parse_die(lex: Peekable[Token]) -> SingleDie:
  """
  Die ->
    | d Number
    | d { Faces (, Faces)* }
    | d% Number
    | df
    | c
  """
  match next(lex, None):
    case Simple(Simple.Kind.D, left):
      match next(lex, None):
        case Number(value, right):
          span = slice(left.start, right.stop)
          if value == 0:
            raise ZeroFacedDie(span)
          elif value == 1:
            return Const(1, span)
          elif value > np.iinfo(np.int64).max:
            raise TooManyFaces(span)
          else:
            return DX(value, span)
        case Simple(Simple.Kind.LBRACE, _):
          face_count = 0
          arrs = []
          while True:
            faces = parse_faces(lex)
            if left is None:
              left = faces.span
            right = faces.span
            face_count += faces.count()
            if face_count > faces_limit:
              raise TooManyFaces(slice(left.start, right.stop))
            arrs.append(faces.to_array())
            match next(lex, None):
              case Simple(Simple.Kind.COMMA, _):
                pass
              case Simple(Simple.Kind.RBRACE, right):
                span = slice(left.start, right.stop)
                if face_count == 0:
                  raise ZeroFacedDie(span)
                elif face_count == 1:
                  return Const(int(arrs[0][0]), span)
                else:
                  return DFaces(np.concat(arrs), span)
              case tok:
                if tok is not None:
                  lex.put_back(tok)
                raise ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACE), tok)
        case tok:
          if tok is not None:
            lex.put_back(tok)
          raise ExpectError((Number, Simple.Kind.LBRACE), tok)
    case Simple(Simple.Kind.DMOD, left):
      match next(lex):
        case Number(value, right):
          span = slice(left.start, right.stop)
          if value == 0:
            return Const(0, span)
          elif value > np.iinfo(np.int64).max:
            raise TooManyFaces(span)
          else:
            return DModX(value, span)
        case tok:
          if tok is not None:
            lex.put_back(tok)
          raise ExpectError(Number, tok)
    case Simple(Simple.Kind.DF, span):
      return DF(span)
    case Simple(Simple.Kind.C, span):
      return Coin(span)
    case tok:
      if tok is not None:
        lex.put_back(tok)
      raise ExpectError(SingleDie, tok)


@dataclass(frozen=True)
class InfiniteFaces(Exception):
  span: slice


def parse_faces(lex: Peekable[Token]) -> Faces:
  """
  Faces -> SignedNumber (.. SignedNumber)? (.. SignedNumber)? (: Number)?
  """
  try:
    match parse_signed_number(lex):
      case Const(start, left):
        pass
  except ExpectError as ex:
    if ex.expected == Const:
      raise ExpectError(Faces, ex.got)
    else:
      raise
  right = left
  match lex.peek(None):
    case Simple(Simple.Kind.RANGE, _):
      next(lex)
      match parse_signed_number(lex):
        case Const(stop, right):
          pass
      match lex.peek(None):
        case Simple(Simple.Kind.RANGE, _):
          next(lex)
          step = stop
          match parse_signed_number(lex):
            case Const(stop, right):
              pass
        case _:
          step = None
    case _:
      step = None
      stop = None
  match lex.peek(None):
    case Simple(Simple.Kind.COLON, _):
      next(lex)
      match next(lex, None):
        case Number(repeat, right):
          pass
        case tok:
          if tok is not None:
            lex.put_back(tok)
          raise ExpectError(Number, tok)
    case _:
      repeat = 1
  span = slice(left.start, right.stop)
  if stop is None or stop == start:
    return Faces(start, None, 1, repeat, span)
  else:
    sign = int(np.sign(stop - start))
    stop = stop + sign
    if step is None:
      step = sign
    else:
      step -= start
    if step * sign <= 0:
      raise InfiniteFaces(span)
    return Faces(start, stop, step, repeat, span)


def parse_cond(lex: Peekable[Token], prev_prec: int = 0) -> Cond:
  """
  Cond -> OrCond
  OrCond ->
    | OrCond "|" OrCond
    | OrCond ^ OrCond
    | AndCond
  AndCond ->
    | AndCond & AndCond
    | AtomCond

  :raises: ExpectError(Cond), ExpectError(Simple.Kind.RPAR), ExpectError(Number)
  """
  lhs = parse_atom_cond(lex)
  while isinstance(tok := lex.peek(None), Simple):
    try:
      op = BinaryCond.Operator[tok.kind.name]
      prec = op.prec()
      if prec <= prev_prec:
        break
    except KeyError:
      break
    next(lex)
    rhs = parse_cond(lex, prec)
    lhs = BinaryCond(op, lhs, rhs, slice(lhs.span.start, rhs.span.stop))
  return lhs


def parse_atom_cond(lex: Peekable[Token]) -> Cond:
  """
  AtomCond ->
    | = SignedNumber
    | > SignedNumber
    | >= SignedNumber
    | < SignedNumber
    | <= SignedNumber
    | odd | even
    | max | min
    | uniq
    | dupmax
    | dup Number?
    | h Number?
    | l Number?
    | ( Cond )
    | ~ AtomCond
    | SignedNumber

  :raises: ExpectError(Cond), ExpectError(Simple.Kind.RPAR), ExpectError(Number)
  """
  match lex.peek(None):
    case Simple(kind, left):
      try:
        comp = Compare.Comparator[kind.name]
        next(lex)
        to = parse_signed_number(lex)
        return Compare(comp, to.value, slice(left.start, to.span.stop))
      except KeyError:
        pass
      try:
        kind = AtomCond.Kind[kind.name]
        next(lex)
        return AtomCond(kind, left)
      except KeyError:
        pass
      match kind:
        case Simple.Kind.DUP:
          kind = CountingCond.Kind.DUP
        case Simple.Kind.H:
          kind = CountingCond.Kind.HIGH
        case Simple.Kind.L:
          kind = CountingCond.Kind.LOW
        case _:
          kind = None
      if kind is not None:
        next(lex)
        match lex.peek(None):
          case Number(count, right):
            next(lex)
            return CountingCond(kind, count, slice(left.start, right.stop))
          case _:
            return CountingCond(kind, None, left)
  match lex.peek(None):
    case Simple(Simple.Kind.LPAR, left):
      next(lex)
      cond = parse_cond(lex)
      match lex.peek(None):
        case Simple(Simple.Kind.RPAR, right):
          next(lex)
          return replace(cast(Any, cond), span=slice(left.start, right.stop))
        case tok:
          raise ExpectError(Simple.Kind.RPAR, tok)
    case Simple(Simple.Kind.NOT, left):
      next(lex)
      cond = parse_atom_cond(lex)
      return Not(cond, slice(left.start, cond.span.stop))
    case _:
      try:
        to = parse_signed_number(lex)
      except ExpectError as ex:
        if ex.expected == Const:
          raise ExpectError(Cond, ex.got)
        else:
          raise
      return Compare(Compare.Comparator.EQ, to.value, to.span)


def parse_signed_number(lex: Peekable[Token]) -> Const:
  """
  SignedNumber ->
    | + Number
    | - Number
    | Number

  :raises: ExpectError(SignedNumber), ExpectError(Number)
  """
  match next(lex, None):
    case Simple(Simple.Kind.PLUS, left):
      sign = 1
    case Simple(Simple.Kind.MINUS, left):
      sign = -1
    case Number(value, span):
      return Const(value, span)
    case tok:
      if tok is not None:
        lex.put_back(tok)
      raise ExpectError(Const, tok)
  match next(lex, None):
    case Number(value, right):
      return Const(sign * value, slice(left.start, right.stop))
    case tok:
      if tok is not None:
        lex.put_back(tok)
      raise ExpectError(Number, tok)
