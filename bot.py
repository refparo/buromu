import nonebot as nb
from nonebot.adapters.console import Adapter as ConsoleAdapter
from nonebot.adapters.onebot.v11 import Adapter as OB11Adapter

nb.init()

driver = nb.get_driver()

if driver.env == "prod":
  driver.register_adapter(OB11Adapter)
else:
  driver.register_adapter(ConsoleAdapter)

nb.load_plugins("plugins")

if __name__ == "__main__":
  nb.run()
