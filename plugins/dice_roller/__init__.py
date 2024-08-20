import os
import textwrap
import traceback
from typing import Final, assert_never, cast

import nonebot as nb
import nonebot.adapters as nba
import nonebot.adapters.onebot.v11 as ob11
import nonebot.params as nbp
import numpy as np
from pydantic import BaseModel

from dice.dice import Const
from dice.expr import (
  Cond,
  Evaluatable,
  Evaluated,
  Rollable,
  RollContext,
  TooManyDice,
  TooManyRerolls,
)
from dice.formattable import FormatContext
from dice.parse import (
  ExpectError,
  Faces,
  InfiniteFaces,
  TooManyFaces,
  ZeroFacedDie,
  parse_expr,
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


class Config(BaseModel):
  bot_uin: str
  line_limit: int = 80
  msg_limit: int = 200
  rep_limit: int = 20


config: Final = nb.get_plugin_config(Config)

gen = np.random.Generator(np.random.SFC64())

full_help_res_id: str | None = None

on_roll = nb.on_command("roll", aliases={"r"}, block=True)


@on_roll.handle()
async def roll(
  bot: nba.Bot, evt: nba.Event, arg: nba.Message = nbp.CommandArg()
):
  global full_help_res_id
  if not all(seg.is_text() for seg in arg):
    await on_roll.finish("不要在掷骰命令里夹杂非文本内容！需要帮助请用 .r -h")
  # begin parse args
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
  # end parse args
  match parsed:
    case RollArgs() as args:
      if args.repeat == 0:
        await on_roll.finish("重复 0 次，意思就是什么都不用做咯？")
      if args.repeat > config.rep_limit:
        await on_roll.finish(
          "想让我多陪一会可以直说，没必要用这么多次掷骰拖时间~"
        )
      if args.analyze or args.hist:
        await on_roll.finish("统计分析功能暂时未实现，敬请期待")
      lex = Peekable(Lexer(args.expr.lower()))
      # begin parse_expr
      try:
        expr = parse_expr(lex)
      except ExpectError as ex:
        match ex.got:
          case None:
            error_msg = "表达式不完整！末尾缺少"
          case Simple(_, span) | Number(_, span) | Unknown(span):
            error_msg = "表达式中有错误！"
            lookback_limit = 10
            if span.start == 0:
              error_msg += "开头"
            elif span.start <= lookback_limit:
              error_msg += f"“{args.expr[:span.start]}”后面"
            else:
              error_msg += (
                f"“…{args.expr[span.start - lookback_limit:span.start]}”后面"
              )
            error_msg += f"不应该是“{args.expr[span]}”，而应该接"
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
      except ZeroFacedDie as ex:
        await on_roll.finish(
          f"“{args.expr[ex.span]}”坍缩成黑洞吞噬了你，你死了"
        )
      except InfiniteFaces as ex:
        await on_roll.finish(
          f"“{args.expr[ex.span]}”展开成的无穷个面把你炸成了碎片"
        )
      except TooManyFaces as ex:
        await on_roll.finish(
          f"“{args.expr[ex.span]}”中巨量的面淹没了你，你淹死了"
        )
      except Exception as ex:
        await on_roll.finish(
          "表达式解析过程中发生了意料之外的异常：\n"
          + "".join(traceback.format_exception(ex)).replace(os.getcwd(), ".")
        )
      # end parse_expr
      if args.repeat == 1:
        roll_ctx = RollContext(gen, [])
      else:
        roll_ctx = RollContext(gen, [args.repeat])
      # begin eval
      steps = [expr]
      del expr
      try:
        while not isinstance(steps[-1], Evaluated):
          steps.append(steps[-1].eval(roll_ctx, not args.steps))
      except TooManyDice as ex:
        await on_roll.finish(
          f"“{args.expr[ex.span]}”产生的巨量骰子充满了整个房间，你被挤压而死"
        )
      except TooManyRerolls as ex:
        await on_roll.finish(f"“{args.expr[ex.span]}”的重骰次数太多了！")
      except Exception as ex:
        await on_roll.finish(
          "掷骰和运算过程中发生了意料之外的异常：\n"
          + "".join(traceback.format_exception(ex)).replace(os.getcwd(), ".")
        )
      # end eval
      # begin compose message
      match lex.peek(None):
        case None:
          note = ""
        case tok:
          note = args.expr[tok.span.start :].strip()
      msg: list[nba.Message | str]
      if isinstance(evt, ob11.GroupMessageEvent):
        at_user = ob11.MessageSegment.at(evt.get_user_id())
        if len(note) != 0:
          msg = [at_user + f" 关于“{note}”的掷骰结果如下"]
        else:
          msg = [at_user + " 的掷骰结果如下"]
      else:
        if len(note) != 0:
          msg = [f"关于“{note}”的掷骰结果如下"]
        else:
          msg = ["掷骰结果如下"]
      if args.repeat == 1:
        msg[-1] += "：\n"
        msg[-1] += compose_message(FormatContext(args.expr, []), steps)
      else:
        msg[-1] += "\n"
        format_ctx = FormatContext(args.expr, [0])
        for i in range(args.repeat):
          format_ctx.select[0] = i
          msg[-1] += f"第 {i + 1} 次掷骰：\n"
          msg[-1] += compose_message(format_ctx, steps)
          if i < args.repeat - 1:
            if len(msg[-1]) > config.msg_limit:
              msg.append("")
            else:
              msg[-1] += "\n"
      # end compose message
      for m in msg:
        if isinstance(m, ob11.Message):
          m.reduce()
        await on_roll.send(m)
    case Help(full):
      if full:
        if isinstance(evt, ob11.GroupMessageEvent):
          if full_help_res_id is None:
            res_id = await bot.call_api(
              "send_forward_msg", messages=full_help_forward_msg
            )
            full_help_res_id = res_id
          else:
            res_id = full_help_res_id
          await on_roll.finish(ob11.MessageSegment.forward(res_id))
        else:
          for seg in full_help:
            await on_roll.send(seg)
      else:
        await on_roll.finish(short_help)
    case never:
      assert_never(never)


def compose_message(ctx: FormatContext, steps: list[Evaluatable]):
  total_len = 0
  lines: list[str] = []
  it = iter(steps)
  step = next(it)
  while True:
    fmt = step.format(ctx)
    lines.append(next(fmt))
    while True:
      try:
        lines[-1] += fmt.send(len(lines[-1]) > config.line_limit)
      except StopIteration:
        break
    if len(lines) >= 2 and lines[-1] == lines[-2]:
      lines.pop()
    else:
      total_len += len(lines[-1])
    try:
      step = next(it)
    except StopIteration:
      break
  if total_len < config.line_limit:
    return " = ".join(lines)
  else:
    return "\n= ".join(lines)


def expectation_to_str(exp: ExpectError.Expectation):
  if exp == Evaluatable:
    return "表达式"
  elif exp == Rollable:
    return "骰池"
  elif exp == Faces:
    return "骰面"
  elif exp == Cond:
    return "条件"
  elif exp == Number:
    return "数字"
  elif exp == Const:
    return "带符号数字"
  elif isinstance(exp, Simple.Kind):
    return f"“{exp.value}”"
  else:
    raise ValueError("unexpected expectation")


short_help: Final = textwrap.dedent("""\
  .rd6  掷一个 d6
  .rd20  掷一个 d20
  .r2d6  掷两个 d6，求和
  .rd100  掷一个 d100
  .rd100#<=50  掷一个 d100，判断是否小于等于 50
  .r10*2d10kl-d%10  掷一次 CoC 优势骰
  .rd20+5  掷一个 d20，加上 5
  .r2d20kh  掷两个 d20，取高
  .r2d20kl  掷两个 d20，取低
  .r[d6,d8]kh  掷一个 d6 和一个 d8，取高
  .r4d6dl  掷 4 个 d6，去掉最低值后求和
  .rd20r1  掷一个 d20，出 1 时重骰一次
  .rd20r<11  掷一个 d20，小于 11 时重骰一次
  .rd6!  掷一个 d6，如果出 6 就再骰一个 d6，还出 6 就继续加骰，最后求和
  .r3d6#6  掷 3 个 d6，数出其中 6 的个数
  .r3d6 -a  分析 3d6 的统计性质
  .r3d6 --hist  绘制 3d6 的分布直方图
  .rd20 -r5  重复掷 5 次 d20
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
      - <expr>  取相反数
      <expr> ** <expr>  求幂次（右结合）
      (<expr>)  括号
      <pool>  将骰池 <pool> 内全部骰子掷骰后求和
      <pool>#[cond]  数出骰池 <pool> 内满足条件 [cond] 的骰子数目，默认数出全部骰子
    （……）""",
    """\
    骰池 <pool>：
      <die>  一个骰子
      <±number>  一个永远等于 <±number> 的骰子
      <number><die>  将骰子 <die> 掷 <number> 次
      [<pool>, ...]  将所有骰池 <pool>... 合并为一个骰池
      <pool>k<cond>  在 <pool> 中取所有满足条件 <cond> 的骰子
      <pool>d<cond>  在 <pool> 中去除所有满足条件 <cond> 的骰子
      <pool>r<cond>  在 <pool> 中，将所有满足条件 <cond> 的骰子重骰一次
      <pool>rr<cond>  在 <pool> 中，重骰所有满足条件 <cond> 的骰子，直到条件不再满足为止
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

full_help_forward_msg: Final = [
  {
    "type": "node",
    "data": {
      "name": "Buromu",
      "uin": "1590667818",
      "content": [ob11.MessageSegment.text(seg)],
    },
  }
  for seg in full_help
]
