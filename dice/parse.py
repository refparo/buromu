from dataclasses import dataclass, replace
from typing import TypeAliasType

from utils.peekable import Peekable

from .expr import (
  Binary,
  BinaryCond,
  Coin,
  Combine,
  Compare,
  Cond,
  CountCond,
  DFace,
  DFaces,
  DFate,
  Die,
  DModFace,
  Expr,
  Faces,
  Filter,
  Neg,
  Not,
  Pool,
  Pow,
  Repeat,
  SignedNumber,
  SimpleCond,
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


def parse_expr(lex: Peekable[Token], prev_prec: int = 0) -> Expr[Pool]:
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
    | AtomExpr

  :raises: ExpectError(Expr), ExpectError(Pool), ExpectError(Faces),
    ExpectError(Cond),
    ExpectError((Number, Simple.Kind.LBRACE)),
    ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACE)),
    ExpectError(Simple.Kind.RPAR),
    ExpectError(SignedNumber), ExpectError(Number)
  """
  lhs = parse_atom_expr(lex)
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
    lhs: Expr[Pool] = Binary(op, lhs, rhs, (lhs.span[0], rhs.span[1]))
  return lhs


def parse_atom_expr(lex: Peekable[Token]) -> Expr[Pool]:
  """
  AtomExpr ->
    | - AtomExpr
    | ( Expr )
    | Pool
    | AtomExpr ** Number

  :raises: ExpectError(Expr), ExpectError(Pool), ExpectError(Faces),
    ExpectError(Cond),
    ExpectError((Number, Simple.Kind.LBRACE)),
    ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACE)),
    ExpectError(Simple.Kind.RPAR),
    ExpectError(SignedNumber), ExpectError(Number)
  """
  match lex.peek(None):
    case Simple(Simple.Kind.MINUS, (left, _)):
      next(lex)
      expr = parse_atom_expr(lex)
      lhs = Neg(expr, (left, expr.span[1]))
    case Simple(Simple.Kind.LPAR, (left, _)):
      next(lex)
      expr = parse_expr(lex)
      match lex.peek(None):
        case Simple(Simple.Kind.RPAR, (_, right)):
          next(lex)
          lhs = replace(expr, span=(left, right))
        case tok:
          raise ExpectError(Simple.Kind.RPAR, tok)
    case tok:
      try:
        lhs = parse_pool(lex)
      except ExpectError as ex:
        if ex.expected == Pool:
          raise ExpectError(Expr, tok)
        else:
          raise
  while True:
    match lex.peek(None):
      case Simple(Simple.Kind.POW, _):
        next(lex)
        match lex.peek(None):
          case Number(_, (_, right)) as exp:
            next(lex)
            lhs: Expr[Pool] = Pow(lhs, exp, (lhs.span[0], right))
          case tok:
            raise ExpectError(Number, tok)
      case _:
        break
  return lhs



def parse_pool(lex: Peekable[Token]) -> Pool:
  """
  Pool ->
    | Pool k Cond
    | Pool d Cond
    | Pool r Cond
    | Pool rr Cond
    | Pool ? Cond
    | Pool ! Cond?
    | Pool uniq
    | AtomPool

  :raises: ExpectError(Pool), ExpectError(Faces), ExpectError(Cond),
    ExpectError((Number, Simple.Kind.LBRACE)),
    ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACE)),
    ExpectError(Simple.Kind.RPAR),
    ExpectError(SignedNumber), ExpectError(Number)
  """
  pool = parse_atom_pool(lex)
  while True:
    match lex.peek(None):
      case Simple(Simple.Kind.K, _):
        op = Filter.Operation.KEEP
      case Simple(Simple.Kind.D, _):
        op = Filter.Operation.DROP
      case Simple(Simple.Kind.R, _):
        op = Filter.Operation.REROLL_ONCE
      case Simple(Simple.Kind.RR, _):
        op = Filter.Operation.REROLL
      case Simple(Simple.Kind.QUERY, _):
        op = Filter.Operation.TEST
      case Simple(Simple.Kind.BANG, (_, right)):
        next(lex)
        op = Filter.Operation.EXPLODE
        try:
          cond: Cond = parse_cond(lex)
        except ExpectError as ex:
          if ex.expected == Cond:
            cond: Cond = SimpleCond(SimpleCond.Kind.MAX, (right, right))
          else:
            raise
        pool: Pool = Filter(op, pool, cond, (pool.span[0], cond.span[1]))
        continue
      case Simple(Simple.Kind.UNIQ, (_, right)):
        next(lex)
        pool: Pool = Unique(pool, (pool.span[0], right))
        continue
      case _:
        return pool
    next(lex)
    cond: Cond = parse_cond(lex)
    pool: Pool = Filter(op, pool, cond, (pool.span[0], cond.span[1]))
    # do not remove these type hints: pyright will freeze without them!


def parse_atom_pool(lex: Peekable[Token]) -> Pool:
  """
  AtomPool ->
    | Die
    | SignedNumber
    | Number Die
    | { Pool (, Pool)* }

  :raises: ExpectError(Pool), ExpectError(Faces), ExpectError(Cond),
    ExpectError((Number, Simple.Kind.LBRACE)),
    ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACE)),
    ExpectError(Simple.Kind.RPAR),
    ExpectError(SignedNumber), ExpectError(Number)
  """
  match lex.peek(None):
    case Simple(Simple.Kind.PLUS | Simple.Kind.MINUS, _):
      return parse_signed_number(lex)
    case Number(_, (left, _) as span) as number:
      next(lex)
      try:
        die = parse_die(lex)
        return Repeat(die, number, (left, die.span[1]))
      except ExpectError as ex:
        if ex.expected == Die:
          return SignedNumber(False, number, span)
        else:
          raise
    case Simple(Simple.Kind.LBRACE, (left, _)):
      next(lex)
      pools: list[Pool] = [parse_pool(lex)]
      while True:
        match lex.peek(None):
          case Simple(Simple.Kind.COMMA, _):
            next(lex)
            pools.append(parse_pool(lex))
          case Simple(Simple.Kind.RBRACE, (_, right)):
            next(lex)
            return Combine(pools, (left, right))
          case tok:
            raise ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACE), tok)
    case tok:
      try:
        return parse_die(lex)
      except ExpectError as ex:
        if ex.expected == Die:
          raise ExpectError(Pool, tok)
        else:
          raise


def parse_die(lex: Peekable[Token]) -> Die:
  """
  Die ->
    | d Number
    | d { Faces (, Faces)* }
    | d% Number
    | df
    | c

  :raises: ExpectError(Die), ExpectError(Faces),
    ExpectError(SignedNumber), ExpectError(Number),
    ExpectError((Number, Simple.Kind.LBRACE)),
    ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACE))
  """
  match lex.peek(None):
    case Simple(Simple.Kind.D, (left, _)):
      next(lex)
      match lex.peek(None):
        case Number(_, (_, right)) as number:
          next(lex)
          return DFace(number, (left, right))
        case Simple(Simple.Kind.LBRACE, _):
          next(lex)
          faces = [parse_faces(lex)]
          while True:
            match lex.peek(None):
              case Simple(Simple.Kind.COMMA, _):
                next(lex)
                faces.append(parse_faces(lex))
              case Simple(Simple.Kind.RBRACE, (_, right)):
                next(lex)
                return DFaces(faces, (left, right))
              case tok:
                raise ExpectError((Simple.Kind.COMMA, Simple.Kind.RBRACE), tok)
        case tok:
          raise ExpectError((Number, Simple.Kind.LBRACE), tok)
    case Simple(Simple.Kind.DMOD, (left, _)):
      next(lex)
      match lex.peek(None):
        case Number(_, (_, right)) as number:
          next(lex)
          return DModFace(number, (left, right))
        case tok:
          raise ExpectError(Number, tok)
    case Simple(Simple.Kind.DF, span):
      next(lex)
      return DFate(span)
    case Simple(Simple.Kind.C, span):
      next(lex)
      return Coin(span)
    case tok:
      raise ExpectError(Die, tok)


def parse_faces(lex: Peekable[Token]) -> Faces:
  """
  Faces -> SignedNumber (.. SignedNumber)? (.. SignedNumber)? (: Number)?

  :raises: ExpectError(Faces), ExpectError(SignedNumber), ExpectError(Number)
  """
  try:
    start = parse_signed_number(lex)
  except ExpectError as ex:
    if ex.expected == SignedNumber:
      raise ExpectError(Faces, ex.got)
    else:
      raise
  right = start.span[1]
  match lex.peek(None):
    case Simple(Simple.Kind.RANGE, _):
      next(lex)
      stop = parse_signed_number(lex)
      right = stop.span[1]
      match lex.peek(None):
        case Simple(Simple.Kind.RANGE, _):
          next(lex)
          step = stop
          stop = parse_signed_number(lex)
          right = stop.span[1]
        case _:
          step = None
    case _:
      step = None
      stop = None
  match lex.peek(None):
    case Simple(Simple.Kind.COLON, _):
      next(lex)
      match lex.peek(None):
        case Number(_, (_, right)) as number:
          next(lex)
          repeat = number
        case tok:
          raise ExpectError(Number, tok)
    case _:
      repeat = None
  return Faces(start, step, stop, repeat, (start.span[0], right))


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
    lhs = BinaryCond(op, lhs, rhs, (lhs.span[0], rhs.span[1]))
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
    case Simple(kind, (left, _) as span):
      try:
        comp = Compare.Comparator[kind.name]
        next(lex)
        to = parse_signed_number(lex)
        return Compare(comp, to, (left, to.span[1]))
      except KeyError:
        pass
      try:
        kind = SimpleCond.Kind[kind.name]
        next(lex)
        return SimpleCond(kind, span)
      except KeyError:
        pass
      match kind:
        case Simple.Kind.DUP:
          kind = CountCond.Kind.DUP
        case Simple.Kind.H:
          kind = CountCond.Kind.HIGH
        case Simple.Kind.L:
          kind = CountCond.Kind.LOW
        case _:
          kind = None
      if kind is not None:
        next(lex)
        match lex.peek(None):
          case Number(_, (_, right)) as count:
            next(lex)
            return CountCond(kind, count, (left, right))
          case _:
            return CountCond(kind, None, span)
  match lex.peek(None):
    case Simple(Simple.Kind.LPAR, (left, _)):
      next(lex)
      cond = parse_cond(lex)
      match lex.peek(None):
        case Simple(Simple.Kind.RPAR, (_, right)):
          next(lex)
          return replace(cond, span=(left, right))
        case tok:
          raise ExpectError(Simple.Kind.RPAR, tok)
    case Simple(Simple.Kind.NOT, (left, _)):
      next(lex)
      cond = parse_atom_cond(lex)
      return Not(cond, (left, cond.span[1]))
    case _:
      try:
        to = parse_signed_number(lex)
      except ExpectError as ex:
        if ex.expected == SignedNumber:
          raise ExpectError(Cond, ex.got)
        else:
          raise
      return Compare(Compare.Comparator.EQ, to, to.span)


def parse_signed_number(lex: Peekable[Token]) -> SignedNumber:
  """
  SignedNumber ->
    | + Number
    | - Number
    | Number

  :raises: ExpectError(SignedNumber), ExpectError(Number)
  """
  match lex.peek(None):
    case Simple(Simple.Kind.PLUS, (left, _)):
      next(lex)
      sign = False
    case Simple(Simple.Kind.MINUS, (left, _)):
      next(lex)
      sign = True
    case Number(_, span) as number:
      next(lex)
      return SignedNumber(False, number, span)
    case tok:
      raise ExpectError(SignedNumber, tok)
  match lex.peek(None):
    case Number(_, (_, right)) as number:
      next(lex)
      return SignedNumber(sign, number, (left, right))
    case tok:
      raise ExpectError(Number, tok)
