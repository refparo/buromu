package one.paro.buromu

import net.mamoe.mirai.console.plugin.jvm.JvmPluginDescription
import net.mamoe.mirai.console.plugin.jvm.KotlinPlugin

object Main: KotlinPlugin(JvmPluginDescription.loadFromResource()) {
  override fun onEnable() {
    Throttler.onEnable()
    Alarm.onEnable()
  }

  override fun onDisable() {
    Alarm.onDisable()
  }
}
