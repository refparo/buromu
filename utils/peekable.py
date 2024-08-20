from collections.abc import Iterator
from typing import cast, overload


class Peekable[T]:
  inner: Iterator[T]
  has_next: bool
  next: T | None

  def __init__(self, inner: Iterator[T]):
    self.inner = inner
    self.has_next = False
    self.next = None

  def __iter__(self):
    return self

  def __next__(self):
    if self.has_next:
      result = self.next
      self.has_next = False
      self.next = None
      return cast(T, result)
    else:
      return self.inner.__next__()

  def put_back(self, this: T):
    """
    Note that this is unchecked: if this `Peekable` already has a `next`,
    it will be silently discarded!

    One should only use this after a call of `next`, before any call of `peek`s.
    """
    self.has_next = True
    self.next = this

  @overload
  def peek(self) -> T: ...
  @overload
  def peek[U](self, default: U, /) -> T | U: ...
  def peek(self, *args):
    if self.has_next:
      return cast(T, self.next)
    else:
      try:
        self.next = next(self.inner)
      except StopIteration:
        if args == ():
          raise
        else:
          return args[0]
      self.has_next = True
      return self.next
