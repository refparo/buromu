from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, override


@dataclass
class FormatContext:
  orig: str
  select: list[int]


class Formattable(ABC):
  @abstractmethod
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]: ...


class TriviallyFormattable(Formattable, ABC):
  span: slice

  @override
  def format(self, ctx: FormatContext) -> Generator[str, bool, None]:
    yield ctx.orig[self.span]
