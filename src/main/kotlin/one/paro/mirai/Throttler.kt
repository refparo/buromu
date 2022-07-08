package one.paro.mirai

import kotlin.math.min
import kotlinx.coroutines.*

import net.mamoe.mirai.console.plugin.jvm.JvmPluginDescription
import net.mamoe.mirai.console.plugin.jvm.KotlinPlugin
import net.mamoe.mirai.contact.User
import net.mamoe.mirai.event.ConcurrencyKind
import net.mamoe.mirai.event.EventPriority
import net.mamoe.mirai.event.events.*
import net.mamoe.mirai.event.globalEventChannel

import one.paro.buromu.sendMessageIn

object Throttler: KotlinPlugin(JvmPluginDescription(
  id = "one.paro.throttler",
  version = "0.0.1",
  block = {
    name("Throttler")
    author("Paro")
    info("通用的消息限流插件")
  }
)) {
  override fun onEnable() {
    limitSimultaneousConversations()
    limitMsgSendRate()
  }

  private val conversations = mutableMapOf<User, Pair<Long, Job>>()
  private val blocked = mutableMapOf<User, Job>()
  private fun limitSimultaneousConversations() {
    globalEventChannel().subscribeAlways<UserMessageEvent>(
      priority = EventPriority.HIGHEST
    ) {
      if (conversations.containsKey(subject))
        conversations[subject]!!.second.cancel()
      if (!conversations.containsKey(subject) && conversations.size >= 3) {
        logger.verbose("blocking a message from ${subject.nick}")
        intercept()
        if (!blocked.containsKey(subject)) {
          logger.verbose("${subject.nick} will be blocked for 10 min")
          subject.sendMessageIn(
            "我在多线聊天！能等会再找我吗？",
            "多线聊天啦！等会再来行嘛！"
          )
          blocked[subject] = launch(coroutineContext) {
            delay(600000)
            blocked.remove(subject)
          }
        }
      } else {
        logger.verbose("establishing conversation with ${subject.nick}")
        val now = System.currentTimeMillis()
        conversations[subject] = Pair(now, launch(coroutineContext) {
          delay(10000)
          if (conversations[subject]!!.first == now)
            conversations.remove(subject)
        })
      }
    }
  }

  private fun limitMsgSendRate() {
    globalEventChannel().subscribeAlways<MessagePreSendEvent>(
      concurrency = ConcurrencyKind.LOCKED
    ) {
      delay(min(message.contentToString().length * 333L, 20000))
    }
  }
}
