package one.paro.buromu

import kotlin.reflect.KClass
import kotlinx.coroutines.*

import net.mamoe.mirai.contact.User
import net.mamoe.mirai.event.events.UserMessageEvent
import net.mamoe.mirai.event.globalEventChannel
import net.mamoe.mirai.event.nextEvent
import net.mamoe.mirai.message.MessageReceipt
import net.mamoe.mirai.utils.MiraiLogger

fun MiraiLogger.derive(requester: KClass<*>): MiraiLogger =
  MiraiLogger.Factory.create(
    requester, this.identity + "." + requester.simpleName)

suspend fun User.sendMessageIn(
  vararg messages: String
): MessageReceipt<User> = sendMessage(messages.random())

fun CoroutineScope.nextMessageAsync(from: User) =
  async(coroutineContext) {
    Core.globalEventChannel()
      .nextEvent<UserMessageEvent> { it.subject == from }
  }
