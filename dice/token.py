import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np

type Token = Simple | Number | Unknown


@dataclass(frozen=True)
class Simple:
  class Kind(Enum):
    PLUS = "+"
    MINUS = "-"
    MULT = "*"
    DIV = "/"
    MOD = "%"
    POW = "**"
    D = "d"
    DMOD = "d%"
    DF = "df"
    C = "c"
    K = "k"
    R = "r"
    RR = "rr"
    BANG = "!"
    COUNT = "#"
    UNIQ = "uniq"
    COMMA = ","
    COLON = ":"
    RANGE = ".."
    EQ = "="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    ODD = "odd"
    EVEN = "even"
    MAX = "max"
    MIN = "min"
    DUP = "dup"
    DUPMAX = "dupmax"
    H = "h"
    L = "l"
    AND = "&"
    OR = "|"
    XOR = "^"
    NOT = "~"
    LPAR = "("
    RPAR = ")"
    LBRACKET = "["
    RBRACKET = "]"
    LBRACE = "{"
    RBRACE = "}"

  kind: Kind
  span: slice


@dataclass(frozen=True)
class Number:
  value: np.int64
  span: slice


@dataclass(frozen=True)
class Unknown:
  span: slice


@dataclass(frozen=True)
class NumberOverflow(Exception):
  span: slice


possible_lengths: Final = sorted(
  {len(kind.value) for kind in Simple.Kind}, reverse=True
)
space_regex: Final = re.compile(r"\s*")
number_regex: Final = re.compile(r"\d+")


class Lexer:
  source: str
  cursor: int

  def __init__(self, source: str):
    self.source = source
    self.cursor = 0

  def __iter__(self):
    return self

  def __next__(self) -> Token:
    if self.cursor >= len(self.source):
      raise StopIteration
    if (spaces := space_regex.match(self.source, self.cursor)) is not None:
      self.cursor = spaces.end()
      if self.cursor >= len(self.source):
        raise StopIteration
    if (number := number_regex.match(self.source, self.cursor)) is not None:
      self.cursor = number.end()
      span = slice(*number.span())
      try:
        return Number(np.int64(number.group()), span)
      except OverflowError:
        raise NumberOverflow(span)
    remaining_length = len(self.source) - self.cursor
    for length in possible_lengths:
      if length > remaining_length:
        continue
      try:
        kind = Simple.Kind(self.source[self.cursor : self.cursor + length])
        break
      except ValueError:
        pass
    else:
      pos = self.cursor
      self.cursor += 1
      return Unknown(slice(pos, self.cursor))
    span = slice(self.cursor, self.cursor + length)
    self.cursor = span.stop
    return Simple(kind, span)
