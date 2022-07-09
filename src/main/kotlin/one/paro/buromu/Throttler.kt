package one.paro.buromu

import kotlin.math.min
import kotlinx.coroutines.*

import net.mamoe.mirai.contact.User
import net.mamoe.mirai.event.ConcurrencyKind
import net.mamoe.mirai.event.EventPriority
import net.mamoe.mirai.event.events.*
import net.mamoe.mirai.event.globalEventChannel

class Throttler: CoroutineScope {
  override val coroutineContext = Core.coroutineContext
  private val logger = Core.logger.derive(this::class)

  fun onEnable() {
    limitSimultaneousConversations()
    limitMsgSendRate()
  }

  private val conversations = mutableMapOf<User, Pair<Long, Job>>()
  private val blocked = mutableMapOf<User, Job>()
  // TODO make a waiting queue instead of directly blocking
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
