package one.paro.buromu

import net.mamoe.mirai.console.MiraiConsole
import net.mamoe.mirai.console.plugin.PluginManager.INSTANCE.enable
import net.mamoe.mirai.console.plugin.PluginManager.INSTANCE.load
import net.mamoe.mirai.console.terminal.MiraiConsoleTerminalLoader
import net.mamoe.mirai.console.util.ConsoleExperimentalApi

import one.paro.mirai.Throttler

@OptIn(ConsoleExperimentalApi::class)
suspend fun main() {
  MiraiConsoleTerminalLoader.startAsDaemon()
  Throttler.load()
  Throttler.enable()
  Core.load()
  Core.enable()
  MiraiConsole.job.join()
}
