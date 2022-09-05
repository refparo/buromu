package one.paro.buromu

import kotlin.reflect.KClass
import kotlinx.coroutines.*

import net.mamoe.mirai.contact.User
import net.mamoe.mirai.event.MessageSubscribersBuilder
import net.mamoe.mirai.event.events.MessageEvent
import net.mamoe.mirai.event.events.UserMessageEvent
import net.mamoe.mirai.event.globalEventChannel
import net.mamoe.mirai.event.nextEvent
import net.mamoe.mirai.message.MessageReceipt
import net.mamoe.mirai.message.data.At
import net.mamoe.mirai.utils.MiraiLogger

fun MiraiLogger.derive(requester: KClass<*>): MiraiLogger =
  MiraiLogger.Factory.create(
    requester, this.identity + "." + requester.simpleName)

suspend fun User.sendMessageIn(
  vararg messages: String
): MessageReceipt<User> = sendMessage(messages.random())

fun CoroutineScope.nextMessageAsync(sender: User) =
  async(coroutineContext) {
    globalEventChannel()
      .nextEvent<UserMessageEvent> { it.sender == sender }
  }
fun CoroutineScope.nextMessageInAsync(subject: User) =
  async(coroutineContext) {
    globalEventChannel()
      .nextEvent<UserMessageEvent> { it.subject == subject }
  }

fun <M: MessageEvent, Ret> MessageSubscribersBuilder<M, Ret, Unit, *>
  .atBotAndMatching(regex: Regex, onEvent: suspend M.(MatchResult) -> Unit) =
  atBot {
    if (message[0] is At)
      regex.matchEntire(
        message.subList(1, message.size)
          .joinToString { it.contentToString() }
      )?.let { onEvent(it) }
  }
