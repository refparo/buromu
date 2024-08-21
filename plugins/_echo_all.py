import nb

on_msg = nb.on_message(block=True)


@on_msg.handle()
async def echo(msg: nb.Message = nb.params.EventMessage()):
  await on_msg.finish(msg)
