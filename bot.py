import nonebot as nb
from nonebot.adapters.console import Adapter as ConsoleAdapter

nb.init()

driver = nb.get_driver()
driver.register_adapter(ConsoleAdapter)

nb.load_plugins("plugins")

if __name__ == "__main__":
  nb.run()
