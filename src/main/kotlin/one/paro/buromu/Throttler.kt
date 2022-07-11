package one.paro.buromu

import kotlin.math.min
import kotlinx.coroutines.*

import net.mamoe.mirai.event.ConcurrencyKind
import net.mamoe.mirai.event.events.MessagePreSendEvent
import net.mamoe.mirai.event.globalEventChannel

object Throttler: CoroutineScope by Main {
  fun onEnable() {
    globalEventChannel().subscribeAlways<MessagePreSendEvent>(
      concurrency = ConcurrencyKind.LOCKED
    ) {
      delay(min(message.contentToString().length * 200L, 20000))
    }
  }
}
