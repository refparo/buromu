import numpy as np

from .expr import Binary, Expr, Neg, Pow
from .token import Number
from .filter import FlatDice


def sum_flat_dice(dice: FlatDice) -> FlatDice:
  return FlatDice(np.sum(dice.data, axis=-1, keepdims=True), [], dice.span)


def evaluate_expr(expr: Expr[FlatDice]) -> FlatDice:
  match expr:
    case Binary(op, lhs, rhs, span):
      lhs = evaluate_expr(lhs)
      rhs = evaluate_expr(rhs)
      match op:
        case Binary.Operator.PLUS:
          return FlatDice(lhs.data + rhs.data, [], span)
        case Binary.Operator.MINUS:
          return FlatDice(lhs.data - rhs.data, [], span)
        case Binary.Operator.MULT:
          return FlatDice(lhs.data * rhs.data, [], span)
        case Binary.Operator.DIV:
          return FlatDice(lhs.data // rhs.data, [], span)
        case Binary.Operator.MOD:
          return FlatDice(lhs.data % rhs.data, [], span)
    case Neg(inner, span):
      inner = evaluate_expr(inner)
      return FlatDice(-inner.data, [], span)
    case Pow(inner, Number(exp, _), span):
      inner = evaluate_expr(inner)
      return FlatDice(inner.data**exp, [], span)
    case arr:
      return arr
