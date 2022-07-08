package one.paro.buromu

import kotlin.reflect.KClass

import net.mamoe.mirai.contact.User
import net.mamoe.mirai.event.events.UserMessageEvent
import net.mamoe.mirai.event.globalEventChannel
import net.mamoe.mirai.message.MessageReceipt
import net.mamoe.mirai.utils.MiraiLogger

fun MiraiLogger.derive(requester: KClass<*>): MiraiLogger =
  MiraiLogger.Factory.create(
    requester, this.identity + "." + requester.simpleName)

suspend fun User.sendMessageIn(
  vararg messages: String
): MessageReceipt<User> = sendMessage(messages.random())

fun waitMessage(from: User, handler: suspend UserMessageEvent.() -> Unit) =
  Core.globalEventChannel()
    .filterIsInstance<UserMessageEvent>()
    .filter { it.subject == from }
    .subscribeOnce<UserMessageEvent> { handler() }
