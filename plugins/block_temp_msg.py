import nonebot.adapters.onebot.v11 as ob11

import nb


@nb.msg.event_preprocessor
def block_temp_msg(evt: ob11.PrivateMessageEvent):
  if evt.sub_type != "friend":
    raise nb.ex.IgnoredException("temporary messaging is forbidden")
