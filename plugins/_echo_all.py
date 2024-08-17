import nonebot as nb
import nonebot.adapters as nba
import nonebot.params as nbp

on_msg = nb.on_message(block=True)


@on_msg.handle()
async def echo(msg: nba.Message = nbp.EventMessage()):
  await on_msg.finish(msg)
