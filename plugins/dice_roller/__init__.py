import os
import textwrap
import traceback
from typing import Final, assert_never, cast

import nonebot as nb
import nonebot.adapters as nba
import nonebot.params as nbp
import numpy as np

from dice.evaluate import evaluate_expr, sum_flat_dice
from dice.expr import Cond, Expr, Faces, Pool, SignedNumber, map_expr
from dice.filter import (
  Filterer,
  FlatDice,
  FlatPool,
  FlattenedExprFormatter,
  TooManyRerolls,
  flatten_pool,
)
from dice.parse import ExpectError, parse_expr
from dice.roll import (
  InfiniteFaces,
  PoolRoller,
  RolledExprFormatter,
  TooManyDice,
  TooManyFaces,
  ZeroFacedDie,
)
from dice.token import Lexer, Number, Simple, Unknown
from utils.peekable import Peekable

from .argparse import (
  DuplicateRepeats,
  Help,
  MalformedRepeatCount,
  MissingDiceExpr,
  MissingRepeatCount,
  RollArgs,
  parse_args,
)

short_help: Final = textwrap.dedent("""
  .rd6  骰一个 d6
  .rd20  骰一个 d20
  .r2d6  骰两个 d6，求和
  .rd100  骰一个 d100
  .rd100?<=50  骰一个 d100，判断是否小于等于 50
  .rd20+5  骰一个 d20，加上 5
  .r2d20kh  骰两个 d20，取高
  .r2d20kl  骰两个 d20，取低
  .r{d6,d8}kh  骰一个 d6 和一个 d8，取高
  .r4d6dl  骰 4 个 d6，去掉最低值后求和
  .rd20r1  骰一个 d20，出 1 时重骰一次
  .rd20r<11  骰一个 d20，小于 11 时重骰一次
  .rd6!  骰一个 d6，如果出 6 就再骰一个 d6，还出 6 就继续加骰，最后求和
  .r3d6?6  骰 3 个 d6，数出其中 6 的个数
  .r3d6 -a  分析 3d6 的统计性质
  .r3d6 --hist  绘制 3d6 的分布直方图
  .rd20 -r5  重复骰 5 次 d20
  若要了解更多高级用法请用 .r --help""")

full_help: Final = [
  textwrap.dedent(seg)
  for seg in (
    """\
    * 布罗姆给你递了一张印着密密麻麻小字的说明书
    指令格式：
      .roll<expr>[note] [options]
      .r<expr>[note] [options]
    （……）""",
    """\
    骰子表达式 <expr>：
      <expr> + <expr>  加法
      <expr> - <expr>  减法
      <expr> * <expr>  乘法
      <expr> / <expr>  除法（向下取整）
      <expr> % <expr>  取模（向下取整）
      <expr> ** <number>  求幂次
      - <expr>  取相反数
      (<expr>)  括号
      <pool>  将骰池 <pool> 内全部骰子掷骰后求和
    （……）""",
    """\
    骰池 <pool>：
      <die>  一个骰子
      <±number>  一个永远等于 <±number> 的骰子
      <number><die>  将骰子 <die> 掷 <number> 次
      {<pool>, ...}  将所有骰池 <pool>... 合并为一个骰池
      <pool>k<cond>  在 <pool> 中取所有满足条件 <cond> 的骰子
      <pool>d<cond>  在 <pool> 中去除所有满足条件 <cond> 的骰子
      <pool>r<cond>  在 <pool> 中，将所有满足条件 <cond> 的骰子重骰一次
      <pool>rr<cond>  在 <pool> 中，重骰所有满足条件 <cond> 的骰子，直到条件不再满足为止
      <pool>?<cond>  在 <pool> 中，将所有满足条件 <cond> 的骰子替换为正面的硬币，不满足的替换为反面的硬币
      <pool>![cond]  在 <pool> 中，将所有满足条件 <cond> 的骰子额外掷一次，若依然满足条件则继续加骰，直到不再满足条件为止；省略条件则默认为 max
      <pool>uniq  在 <pool> 中，对于每种结果只保留一个骰子
    （……）""",
    """\
    骰子 <die>：
      d<number>  一个 <number> 面的骰子，每面分别标有 1, 2, ..., <number>，<number> > 0
      d{<faces>, ...}  一个由骰面 <faces>... 组成的骰子
      d%<number>  一个 <number> 面的骰子，每面分别标有 0, 1, ..., <number> - 1，<number> > 0
      df  一个命运骰（一共六面，其中两面是 -1，两面是 0，两面是 +1）
      c  一枚硬币（一共两面，正面是 1，反面是 0）
    """,
    """\
    骰面 <faces>：
      <±number>  一个标有 <±number> 的面
      <±number1>:<±number2>  把 <±number1> 重复 <number2> 个面
      <±number1>..<±number2>  从 <±number1> 到 <±number2> 的每个数字各一面
      <±number1>..<±number2>:<number3>  从 <±number1> 到 <±number2> 的每个数字各重复 <number3> 面
      <±number1>..<±number2>..<±number3>  从 <±number1> 到 <±number3>，每项间隔 <±number2> - <±number1> 的所有数字各一面
      <±number1>..<±number2>..<±number3>:<number4>  从 <±number1> 到 <±number3>，每项间隔 <±number2> - <±number1> 的所有数字各重复 <number4> 面
    （……）""",
    """\
    条件 <cond>：
      <±number>  等于 <±number>
      = <±number>  等于 <±number>
      > <±number>  大于 <±number>
      >= <±number>  大于等于 <±number>
      < <±number>  小于 <±number>
      <= <±number>  小于等于 <±number>
      odd  是奇数
      even  是偶数
      max  取到了自身的最大值
      min  取到了自身的最小值
      uniq  在骰池内没有重复
      dupmax  在骰池内重复的个数最多
      dup[number]  在骰池内重复了至少 [number] 个，默认为 2
      h[number]  是骰池内最大的 [number] 个之一，默认为 1
      l[number]  是骰池内最小的 [number] 个之一，默认为 1
      <cond> | <cond>  两个条件至少有一个成立
      <cond> ^ <cond>  两个条件有且只有一个成立
      <cond> & <cond>  两个条件同时成立
      ~ <cond>  条件不成立
      (<cond>)  括号
    （……）""",
    """\
    备注 [note]：为掷骰提供的补充说明
    选项 [options]：
      -a, --ana, --analyze  分析表达式的统计性质
      --hist  绘制表达式结果的分布直方图
      -r<number>, --repeat <number>  重复掷骰 <number> 次
      -s, --steps  给出分步运算过程
      -h  获取简短的说明
      --help  获取本说明
    指令中所有字母均不区分大小写""",
  )
]

gen = np.random.Generator(np.random.SFC64())

line_limit: Final = 100
msg_limit: Final = 500

on_roll = nb.on_command("roll", aliases={"r"}, block=True)


@on_roll.handle()
async def roll(arg: nba.Message = nbp.CommandArg()):
  if not all(seg.is_text() for seg in arg):
    await on_roll.finish("不要在掷骰命令里夹杂非文本内容！需要帮助请用 .r -h")
  try:
    parsed = parse_args(arg.extract_plain_text())
  except MissingDiceExpr:
    await on_roll.finish("请问是要让我骰什么？需要讲解格式的话请用 .r -h")
  except MissingRepeatCount as ex:
    await on_roll.finish(
      f"请在 {ex.option} 后面给出需要重复的次数！需要帮助请用 .r -h"
    )
  except MalformedRepeatCount as ex:
    await on_roll.finish(
      f"{ex.option} 后面必须是正确的数字！需要帮助请用 .r -h"
    )
  except DuplicateRepeats as ex:
    await on_roll.finish(
      f"你给出了多个不同的重复次数：{"，".join(ex.options)}。到底以哪个为准？"
    )
  match parsed:
    case RollArgs() as args:
      if args.repeat == 0:
        await on_roll.finish("重复 0 次，意思就是什么都不用做咯？")
      lex = Peekable(Lexer(args.expr.lower()))
      # begin parse_expr
      try:
        expr = parse_expr(lex)
      except ExpectError as ex:
        match ex.got:
          case None:
            error_msg = "表达式不完整！末尾缺少"
          case Simple(_, (pos, _)) | Number(_, (pos, _)) | Unknown(pos) as got:
            error_msg = "表达式中有错误！"
            lookback_limit = 10
            if pos == 0:
              error_msg += "开头"
            elif pos <= lookback_limit:
              error_msg += f"“{args.expr[:pos]}”后面"
            else:
              error_msg += f"“…{args.expr[pos - lookback_limit:pos]}”后面"
            error_msg += "不应该是"
            match got:
              case Simple(kind, _):
                error_msg += f"“{kind.value}”"
              case Number(value, _):
                error_msg += f"“{value}”"
              case Unknown():
                error_msg += f"“{args.expr[pos]}”"
            error_msg += "，而应该接"
        match ex.expected:
          case (*exps, exp_last):
            error_msg += (
              "、".join(expectation_to_str(exp) for exp in exps)
              + "或者"
              + expectation_to_str(exp_last)
            )
          case (exp,) | exp:
            error_msg += expectation_to_str(cast(ExpectError.Expectation, exp))
        await on_roll.finish(error_msg)
      except Exception as ex:
        await on_roll.finish(
          "表达式解析过程中发生了意料之外的异常：\n"
          + "".join(traceback.format_exception(ex)).replace(os.getcwd(), ".")
        )
      # end parse_expr
      if args.repeat == 1:
        shape = []
      else:
        shape = [args.repeat]
      # begin roll_expr
      try:
        rolled = map_expr(expr, PoolRoller(gen, shape).roll_pool)
      except ZeroFacedDie as ex:
        await on_roll.finish(
          args.expr[ex.die.span[0] : ex.die.span[1]]
          + " 坍缩成黑洞吞噬了你，你死了"
        )
      except InfiniteFaces as ex:
        await on_roll.finish(
          args.expr[ex.faces.span[0] : ex.faces.span[1]]
          + " 展开成的无穷个面把你炸成了碎片"
        )
      except TooManyFaces as ex:
        await on_roll.finish(
          args.expr[ex.die.span[0] : ex.die.span[1]]
          + " 中巨量的面淹没了你，你淹死了"
        )
      except TooManyDice:
        await on_roll.finish(
          "骰子表达式产生的巨量骰子充满了整个房间，你被挤压而死"
        )
      except Exception as ex:
        await on_roll.finish(
          "掷骰过程中发生了意料之外的异常：\n"
          + "".join(traceback.format_exception(ex)).replace(os.getcwd(), ".")
        )
      # end roll_expr
      # begin filter & evaluate
      try:
        step_result = map_expr(rolled, lambda pool: flatten_pool(pool, shape))
        steps: list[Expr[FlatPool]] = []
        filterer = Filterer(gen, shape)
        if args.steps:
          while True:
            terminate = True

            def func(pool: FlatPool):
              nonlocal terminate
              result = filterer.filter_step(pool)
              if not isinstance(result, FlatDice):
                terminate = False
              return result

            step_result = map_expr(step_result, func)
            if terminate:
              filtered = cast(Expr[FlatDice], step_result)
              break
            else:
              steps.append(step_result)
          print(len(steps))
        else:
          filtered = map_expr(step_result, filterer.filter_all)
      except TooManyRerolls as ex:
        await on_roll.finish(
          args.expr[ex.span[0] : ex.span[1]] + " 的重骰次数太多了！"
        )
      except Exception as ex:
        await on_roll.finish(
          "运算过程中发生了意料之外的异常：\n"
          + "".join(traceback.format_exception(ex)).replace(os.getcwd(), ".")
        )
      # end filter & evaluate
      summed = map_expr(filtered, sum_flat_dice)
      evaluated = evaluate_expr(summed)
      # compose message
      note = args.expr[expr.span[1] :]
      if note.strip() != "":
        msg = f"关于“{note}”的掷骰结果如下：\n"
      else:
        msg = "掷骰结果如下：\n"
      for i in range(args.repeat):
        if args.repeat > 1:
          msg += f"第 {i + 1} 次掷骰：\n"
          select = [i]
        else:
          select = []
        msg += args.expr[expr.span[0] : expr.span[1]]
        line = "\n= "
        for seg in RolledExprFormatter(select, args.expr).expr_to_str(rolled):
          line += seg
          if len(line) > line_limit:
            line += "…"
            break
        msg += line
        fmt = FlattenedExprFormatter(select, args.expr)
        for step in steps:
          line = "\n= "
          for seg in fmt.expr_to_str(step):
            line += seg
            if len(line) > line_limit:
              line += "…"
              break
          msg += line
        line = "\n= "
        for seg in fmt.expr_to_str(filtered):
          line += seg
          if len(line) > line_limit:
            line += "…"
            break
        msg += line
        line = "\n= "
        for seg in fmt.expr_to_str(summed):
          line += seg
          if len(line) > line_limit:
            line += "…"
            break
        msg += line
        line = "\n= "
        for seg in fmt.expr_to_str(evaluated):
          line += seg
          if len(line) > line_limit:
            line += "…"
            break
        msg += line
        if len(msg) > msg_limit:
          await on_roll.send(msg)
          msg = ""
        else:
          msg += "\n"
      if msg != "":
        await on_roll.finish(msg)
      # end compost message
    case Help(full):
      if full:
        for seg in full_help:
          await on_roll.send(seg)
      else:
        await on_roll.finish(short_help)
    case never:
      assert_never(never)


def expectation_to_str(exp: ExpectError.Expectation):
  if exp == Expr:
    return "表达式"
  elif exp == Pool:
    return "骰池"
  elif exp == Faces:
    return "骰面"
  elif exp == Cond:
    return "条件"
  elif exp == Number:
    return "数字"
  elif exp == SignedNumber:
    return "带符号数字"
  elif isinstance(exp, Simple.Kind):
    return f"“{exp.value}”"
  else:
    raise ValueError("unexpected expectation")
