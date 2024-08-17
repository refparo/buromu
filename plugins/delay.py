import asyncio as aio
from datetime import datetime, timedelta
import math
from typing import Any, Final

import nonebot as nb
import nonebot.adapters as nba
import nonebot.adapters.console as console
import nonebot.adapters.onebot.v11 as ob11
import nonebot.message as nbm
import nonechat.message as ncm
from pydantic import BaseModel


class Config(BaseModel):
  reading_speed_cps: float = 50.0
  typing_speed_kps: float = 10.0


config: Final = nb.get_plugin_config(Config)

next_moment_to_start_typing = datetime.now()


@nbm.event_preprocessor
async def read_msg(event: nba.Event):
  global next_moment_to_start_typing
  if event.get_type() == "message":
    reading_time = len(event.get_plaintext()) / config.reading_speed_cps
    next_moment_to_start_typing = max(
      next_moment_to_start_typing,
      datetime.now() + timedelta(seconds=reading_time),
    )


onebot11_send_msg_api_names: Final = {
  "send_private_msg",
  "send_group_msg",
  "send_msg",
}


@nba.Bot.on_calling_api
async def typing_delay(bot: nba.Bot, api: str, data: dict[str, Any]):
  global next_moment_to_start_typing
  now = datetime.now()
  msg_len: int
  match bot:
    case console.Bot():
      if api != "send_msg":
        return
      match data["message"]:
        case ncm.ConsoleMessage() as msg:
          msg_len = len(str(msg))
        case msg:
          assert False, f"unsupported message type: {type(msg)}"
    case ob11.Bot():
      if api not in onebot11_send_msg_api_names:
        return
      match data["message"]:
        case str(msg):
          msg_len = len(msg)
        case nba.Message() as msg:
          msg_len = len(msg.extract_plain_text())
        case msg:
          assert False, f"unsupported message type: {type(msg)}"
    case _:
      assert False, f"unsupported bot type: {type(bot)}"
  typing_time = math.log(msg_len / config.typing_speed_kps + 1.0)
  nb.logger.debug(
    f"sending a message of length {msg_len}, delaying for {typing_time} s"
  )
  next_moment_to_start_typing += timedelta(seconds=typing_time)
  if next_moment_to_start_typing > now:
    await aio.sleep((next_moment_to_start_typing - now).total_seconds())
  else:
    next_moment_to_start_typing = now
