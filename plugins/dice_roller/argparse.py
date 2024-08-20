from dataclasses import dataclass, replace
import re
from typing import Final


@dataclass
class RollArgs:
  expr: str
  analyze: bool = False
  hist: bool = False
  steps: bool = False
  repeat: int = 1


@dataclass
class Help:
  full: bool


@dataclass
class MissingDiceExpr(Exception):
  pass


@dataclass
class MissingRepeatCount(Exception):
  option: str


@dataclass
class MalformedRepeatCount(Exception):
  option: str


@dataclass
class DuplicateRepeats(Exception):
  options: list[str]


@dataclass
class Break(Exception):
  pass


option_regex: Final = re.compile(r"(?:^|\s+)((--?)([^-]*$))")


def parse_args(args: str):
  """
  :raises: MissingDiceExpr, MissingRepeatCount, MalformedRepeatCount, DuplicateRepeats
  """
  result = RollArgs("")
  dup_repeats = []
  while (m := option_regex.search(args)) is not None:
    (full_option, leader, option) = m.groups()
    if leader == "-":  # short options
      new_result = replace(result)
      try:
        for i, c in enumerate(option):
          match c:
            case "a":
              new_result.analyze = True
            case "s":
              new_result.steps = True
            case "h":
              return Help(full=False)
            case "r":
              rest = option[i + 1 :].strip()
              if len(rest) == 0:
                raise MissingRepeatCount(full_option)
              try:
                new_result.repeat = int(rest)
              except ValueError:
                raise MalformedRepeatCount(leader + option[: i + 1])
              dup_repeats.append(full_option)
              break
            case _:
              raise Break
      except Break:
        break
      result = new_result
    elif leader == "--":  # long options
      match option.rstrip():
        case "ana" | "analyze":
          result.analyze = True
        case "hist":
          result.hist = True
        case "steps":
          result.steps = True
        case "help":
          return Help(full=True)
        case option if option.startswith("repeat"):
          rest = option[6:].strip()
          if len(rest) == 0:
            raise MissingRepeatCount(full_option)
          try:
            result.repeat = int(rest)
          except ValueError:
            raise MalformedRepeatCount("--repeat")
          dup_repeats.append(full_option)
        case _:
          break
    else:
      assert False, "impossible"
    start = m.start()
    args = args[:start]
  result.expr = args.lstrip()
  if len(result.expr) == 0:
    raise MissingDiceExpr
  if len(dup_repeats) > 1:
    dup_repeats.reverse()
    raise DuplicateRepeats(dup_repeats)
  return result
