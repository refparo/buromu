import textwrap
from typing import Final

import nonebot.adapters.onebot.v11 as ob11
import sqlalchemy as sa
from nonebot.matcher import Matcher
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import nb


class Base(DeclarativeBase):
  pass


class EnabledPlugin(Base):
  __tablename__ = "enabled_plugins"
  plugin_id: Mapped[str] = mapped_column(primary_key=True)
  group_id: Mapped[int] = mapped_column(primary_key=True)


engine = sa.create_engine("sqlite+pysqlite:///data/plugin_control.db")

Base.metadata.create_all(engine)


@nb.msg.run_preprocessor
def check_enabled(evt: ob11.GroupMessageEvent, matcher: Matcher):
  if not evt.is_tome() and matcher.plugin_id is not None:
    with engine.connect() as conn:
      if not conn.scalar(
        sa.select(sa.literal(True)).where(
          EnabledPlugin.plugin_id == matcher.plugin_id,
          EnabledPlugin.group_id == evt.group_id,
        )
      ):
        raise nb.ex.IgnoredException(
          f"plugin {matcher.plugin_id} is disabled for group {evt.group_id}"
        )


on_cmd_plugin = nb.on_command(
  "plugin",
  force_whitespace=True,
  permission=nb.perm.SUPERUSER | ob11.permission.GROUP_ADMIN,
)


@on_cmd_plugin.handle()
async def plugin_control(
  bot: nb.Bot,
  evt: ob11.GroupMessageEvent,
  arg: nb.Message = nb.params.CommandArg(),
):
  if not all(seg.is_text() for seg in arg):
    await on_cmd_plugin.finish("不要在命令中夹杂非文本内容！")
  args = arg.extract_plain_text().split()
  match args:
    case ["enable", *plugins]:
      plugins = set(plugins) & {
        plugin.id_ for plugin in nb.get_loaded_plugins()
      }
      if len(plugins) == 0:
        await on_cmd_plugin.finish("未成功启用任何插件")
      with Session(engine) as sess:
        sess.add_all(
          EnabledPlugin(plugin_id=plugin, group_id=evt.group_id)
          for plugin in plugins
        )
        sess.commit()
      await on_cmd_plugin.finish("已在本群启用：" + "、".join(plugins))
    case ["disable", *plugins]:
      with Session(engine) as sess:
        disabled = sess.scalars(
          sa.delete(EnabledPlugin)
          .where(
            EnabledPlugin.plugin_id.in_(plugins),
            EnabledPlugin.group_id == evt.group_id,
          )
          .returning(EnabledPlugin.plugin_id)
        ).all()
        sess.commit()
      if len(disabled) == 0:
        await on_cmd_plugin.finish("未成功禁用任何插件")
      else:
        await on_cmd_plugin.finish("已在本群禁用：" + "、".join(disabled))
    case ["list"]:
      with engine.connect() as conn:
        plugins = conn.scalars(
          sa.select(EnabledPlugin.plugin_id).where(
            EnabledPlugin.group_id == evt.group_id
          )
        ).all()
      if len(plugins) == 0:
        await on_cmd_plugin.finish("本群未启用任何插件")
      else:
        await on_cmd_plugin.finish("本群启用的插件：" + "、".join(plugins))
    case _:
      await on_cmd_plugin.finish(help_msg)


help_msg: Final = textwrap.dedent("""\
  指令格式：
  .plugin enable <plugin...>  --  在本群启用插件
  .plugin disable <plugin...>  --  在本群禁用插件
  .plugin list  --  列出本群启用的插件""")
