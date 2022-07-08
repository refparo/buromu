package one.paro.buromu

import net.mamoe.mirai.console.plugin.jvm.JvmPluginDescription
import net.mamoe.mirai.console.plugin.jvm.KotlinPlugin

object Core: KotlinPlugin(JvmPluginDescription.loadFromResource()) {
  private val alarm = HourAlarm()
  private val throttler = Throttler()

  override fun onEnable() {
    alarm.onEnable()
    throttler.onEnable()
  }

  override fun onDisable() {
    alarm.onDisable()
  }
}
