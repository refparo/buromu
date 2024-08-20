import nonebot.adapters.onebot.v11 as ob11
import nonebot.exception as nbe
import nonebot.message as nbm

@nbm.event_preprocessor
def block_temp_msg(evt: ob11.PrivateMessageEvent):
  if evt.sub_type != "friend":
    raise nbe.IgnoredException("temporary messaging is forbidden")
