package one.paro.buromu

import net.mamoe.mirai.console.plugin.jvm.JvmPluginDescription
import net.mamoe.mirai.console.plugin.jvm.KotlinPlugin

object Core: KotlinPlugin(JvmPluginDescription.loadFromResource()) {
  private val alarm = HourAlarm()

  override fun onEnable() {
    alarm.enable()
  }

  override fun onDisable() {
    alarm.disable()
  }
}
