package one.paro.buromu

import java.util.*
import java.time.LocalTime
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.nanoseconds
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.*

import net.mamoe.mirai.Bot
import net.mamoe.mirai.console.data.AutoSavePluginData
import net.mamoe.mirai.console.data.value
import net.mamoe.mirai.contact.Friend
import net.mamoe.mirai.event.globalEventChannel
import net.mamoe.mirai.event.subscribeFriendMessages

import one.paro.buromu.Core.reload

class HourAlarm: CoroutineScope {
  override val coroutineContext = Core.coroutineContext
  private val logger = Core.logger.derive(this::class)

  private object Data: AutoSavePluginData("Buromu.HourAlarm") {
    val users by value<
      MutableMap<Long,
        MutableMap<Long,
          MutableMap<Int, String>>>>()
  }

  private val timer = Timer()
  private val task = object: TimerTask() {
    fun go(
      fn: suspend CoroutineScope.(Friend, MutableMap<Int, String>) -> Unit
    ) {
      launch(coroutineContext) {
        Data.users.forEach outer@ { (botId, it) ->
          val bot = Bot.getInstanceOrNull(botId)
          if (bot == null) {
            logger.warning("Can't find friend bot $botId")
            return@outer
          }
          it.forEach inner@ { (friendId, activities) ->
            val subject = bot.getFriend(friendId)
            if (subject == null) {
              logger.warning("Can't find friend $friendId of bot $bot")
              return@inner
            }
            fn(subject, activities)
          }
        }
      }
    }
    override fun run() {
      val now = LocalTime.now().hour
      if (now < 9 || now > 21) return
      logger.verbose("Alarm goes off!")
      when (now) {
        9 -> go { subject, activities ->
          activities.clear()
          subject.sendMessageIn(
            "该干活啦！",
            "工作时间到了！"
          )
          subject.sendMessage("这个点不会还没起床吧")
          val listener = waitMessage(subject) {
            subject.sendMessageIn(
              "那就加油吧",
              "今天也要加把劲哦",
              "那就精神满满地开始这一天吧"
            )
          }
          delay(5.minutes)
          if (listener.complete()) {
            subject.sendMessageIn(
              "真没起啊……",
              "真是小懒虫……"
            )
            subject.sendMessageIn(
              "看到这条消息就赶紧起床干活啦",
              "唉，拿你没办法"
            )
          }
        }
        21 -> go { subject, activities ->
          subject.sendMessageIn(
            "一天差不多结束了",
            "已经九点了"
          )
          subject.sendMessageIn(
            "回顾一下这一天吧……",
            "看看这一天你做了些什么……"
          )
          activities.keys.sorted().forEach { time ->
            subject.sendMessage("${time}点，你在${activities[time]}")
          }
          activities.clear()
          subject.sendMessageIn(
            "感觉怎么样？",
            "评价一下自己？"
          )
          val listener = waitMessage(subject) {
            subject.sendMessageIn(
              "之后，差不多该休息了吧？",
              "到休息的时候了"
            )
            subject.sendMessageIn(
              "去洗漱吧",
              "关掉电脑吧"
            )
          }
          delay(5.minutes)
          if (listener.complete()) {
            subject.sendMessage("不管怎样，现在到休息时间了")
            subject.sendMessageIn(
              "去洗漱吧",
              "关掉电脑吧"
            )
          }
        }
        else -> go { subject, activities ->
          subject.sendMessageIn(
            "又一个小时过去了",
            "${now}点到了！"
          )
          subject.sendMessageIn(
            "这一小时做了什么呢",
            "在做什么呢？"
          )
          val listener = waitMessage(subject) {
            activities[now - 1] = message.contentToString()
            if (activities[now - 1]!!
                .contains(Regex("什么都没做|什么也没做|摸鱼"))) {
              activities[now - 1] = "摸鱼"
              subject.sendMessageIn("啊这", "这样啊")
              subject.sendMessage("那接下来得加油咯")
            } else {
              subject.sendMessageIn("不错嘛", "可以")
              subject.sendMessageIn(
                "休息一下再继续吧",
                "继续之前，给自己一点放松时间吧"
              )
            }
          }
          delay(5.minutes)
          if (listener.complete()) {
            subject.sendMessage("还在忙吗……")
            subject.sendMessage("那不打扰你了")
            subject.sendMessage("不过，最好还是休息一下")
          }
        }
      }
    }
  }

  fun enable() {
    Data.reload()

    val now = LocalTime.now()
    timer.scheduleAtFixedRate(task,
      (1.hours
        - now.minute.minutes
        - now.second.seconds
        - now.nano.nanoseconds).inWholeMilliseconds,
      1.hours.inWholeMilliseconds)

    Core.globalEventChannel(coroutineContext).subscribeFriendMessages {
      contains("帮我管理时间") {
        if (it.contains("不用帮我管理时间")) return@contains
        if (Data.users[bot.id]?.containsKey(subject.id) == true) {
          subject.sendMessage("我已经在帮你了")
        } else {
          subject.sendMessageIn("好啊", "好")
          subject.sendMessageIn(
            "不过，既然要求了，就要听话哦",
            "但可不要反悔啊"
          )
          // MutableMap.getOrPut doesn't work here
          if (!Data.users.containsKey(bot.id))
            Data.users[bot.id] = mutableMapOf()
          Data.users[bot.id]!![subject.id] = mutableMapOf()
        }
      }
      contains("不用帮我管理时间") {
        if (Data.users[bot.id]?.containsKey(subject.id) == true) {
          subject.sendMessage("好吧")
          subject.sendMessageIn(
            "希望你没有我帮忙也能管好自己的时间",
            "你能自己管理时间那是最好"
          )
          Data.users[bot.id]?.remove(subject.id)
        } else {
          subject.sendMessage("我怎么不记得你有叫我帮过你")
        }
      }
      matching(Regex("""(\d{1,2})点我在(.+)""")) {
        if (Data.users[bot.id]?.containsKey(subject.id) == true) {
          val (time, activity) = it.destructured
          Data.users[bot.id]!![subject.id]!![time.toInt()] = activity
          subject.sendMessageIn("知道了", "收到")
        } else {
          subject.sendMessage("我怎么不记得你有叫我帮过你")
        }
      }
      matching(Regex("""刚才我在(.+)""")) {
        if (Data.users[bot.id]?.containsKey(subject.id) == true) {
          val (activity) = it.destructured
          Data.users[bot.id]!![subject.id]!![LocalTime.now().hour - 1] =
            activity
          subject.sendMessageIn("知道了", "收到")
        } else {
          subject.sendMessage("我怎么不记得你有叫我帮过你")
        }
      }
    }
  }
  fun disable() {
    timer.cancel()
  }
}
