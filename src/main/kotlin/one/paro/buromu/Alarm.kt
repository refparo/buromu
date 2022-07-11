package one.paro.buromu

import java.util.*
import java.time.LocalDate
import java.time.LocalTime
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.nanoseconds
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.*
import kotlinx.coroutines.selects.select
import kotlinx.serialization.Serializable

import net.mamoe.mirai.Bot
import net.mamoe.mirai.console.data.AutoSavePluginData
import net.mamoe.mirai.console.data.value
import net.mamoe.mirai.contact.Friend
import net.mamoe.mirai.contact.User
import net.mamoe.mirai.event.EventPriority
import net.mamoe.mirai.event.globalEventChannel
import net.mamoe.mirai.event.subscribeFriendMessages

import one.paro.buromu.Main.reload

@OptIn(ExperimentalCoroutinesApi::class)
object Alarm: CoroutineScope by Main {
  private val logger = Main.logger.derive(this::class)

  private object Data: AutoSavePluginData("Buromu.Alarm") {
    private val users by value<MutableMap<Long, MutableMap<Long, UserData>>>()
    operator fun get(user: User) = users[user.bot.id]?.get(user.id)
    operator fun set(user: User, data: UserData) {
      if (!users.containsKey(user.bot.id))
        users[user.bot.id] = mutableMapOf()
      users[user.bot.id]!![user.id] = data
    }
    fun remove(user: User) =
      users[user.bot.id]?.remove(user.id)
    fun contains(user: User) =
      users[user.bot.id]?.containsKey(user.id) == true
    inline fun forEach(action: (Pair<Friend, UserData>) -> Unit) {
      users.forEach outer@ { (botId, it) ->
        val bot = Bot.getInstanceOrNull(botId)
        if (bot == null) {
          logger.warning("Can't find bot $botId")
          return@outer
        }
        it.forEach inner@ { (friendId, activities) ->
          val subject = bot.getFriend(friendId)
          if (subject == null) {
            logger.warning("Can't find friend $friendId of bot $bot")
            return@inner
          }
          action(Pair(subject, activities))
        }
      }
    }
  }
  @Serializable
  private class UserData {
    val tasks: MutableSet<String> = mutableSetOf()
    val activities: MutableMap<Int, String> = mutableMapOf()
    var assessment: String? = null

    val calendar: MutableMap<String, UserDay> = mutableMapOf()

    var needWakeUp = true
    var startOfDay = 9
    var endOfDay = 21
  }
  @Serializable
  private data class UserDay(
    val tasks: Set<String>,
    val activities: Map<Int, String>,
    val assessment: String?
  )

  fun onEnable() {
    Data.reload()

    LocalTime.now().let {
      timer.scheduleAtFixedRate(
        task,
        (1.hours
          - it.minute.minutes
          - it.second.seconds
          - it.nano.nanoseconds).inWholeMilliseconds,
        1.hours.inWholeMilliseconds
      )
    }

    globalEventChannel().subscribeFriendMessages(
      priority = EventPriority.NORMAL
    ) {
      contains("帮我管理时间") {
        if (it.contains(Regex("(不用|不要|别)帮我管理时间"))) return@contains
        if (Data.contains(subject)) {
          subject.sendMessage("我已经在帮你了")
        } else {
          subject.sendMessageIn("好啊", "好")
          subject.sendMessageIn(
            "不过，既然要求了，就要听话哦",
            "但可不要反悔啊"
          )
          Data[subject] = UserData()
        }
      }
      finding(Regex("(不用|不要|别)帮我管理时间")) {
        if (Data.contains(subject)) {
          subject.sendMessage("好吧")
          subject.sendMessageIn(
            "希望你没有我帮忙也能管好自己的时间",
            "你能自己管理时间那是最好"
          )
          Data.remove(subject)
        } else {
          subject.sendMessage("我怎么不记得你有叫我帮过你")
        }
      }
      matching(Regex("""(\d{1,2})点我在(.+)""")) {
        if (Data.contains(subject)) {
          val time = it.groupValues[1].toInt()
          val activity = it.groupValues[2]
          val now = LocalTime.now().hour
          if (time < now) {
            Data[subject]!!.activities[time] = activity
            subject.sendMessageIn("知道了", "收到")
          } else if (time == now) {
            subject.sendMessage("${time}点还没结束呢！")
          } else {
            subject.sendMessage("${time}点还没到呢！")
          }
        }
      }
      matching(Regex("刚才我?在(.+)")) {
        if (Data.contains(subject)) {
          val (activity) = it.destructured
          Data[subject]!!.activities[LocalTime.now().hour - 1] = activity
          subject.sendMessageIn("知道了", "收到")
        }
      }
    }
  }

  private val timer = Timer()
  private val task = object: TimerTask() {
    override fun run() {
      val now = LocalTime.now().hour
      Data.forEach { (subject, data) -> launch { when (now) {
        0 -> {
          data.calendar[LocalDate.now().minusDays(1).toString()] =
            UserDay(data.tasks, data.activities, data.assessment)
          data.tasks.clear()
          data.activities.clear()
          data.assessment = null
          data.calendar[LocalDate.now().toString()]?.let {
            data.tasks.addAll(it.tasks)
          }
        }
        data.startOfDay -> {
          subject.sendMessageIn(
            "该干活啦！",
            "工作时间到了！"
          )
          if (!data.needWakeUp) return@launch
          subject.sendMessage("这个点不会还没起床吧")
          select<Unit> {
            nextMessageAsync(subject).onAwait {
              val msg = it.message.contentToString()
              if (msg.contains(Regex("当然|怎么可能"))) {
                subject.sendMessage("这么自信")
                subject.sendMessage("那我以后就不问了")
                data.needWakeUp = false
              } else {
                subject.sendMessageIn(
                  "那就加油吧",
                  "今天也要加把劲哦",
                  "那就精神满满地开始这一天吧"
                )
              }
            }
            onTimeout(300_000) {
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
        }
        in data.startOfDay until data.endOfDay -> {
          subject.sendMessageIn(
            "又一个小时过去了",
            "${now}点到了！"
          )
          subject.sendMessageIn(
            "这一小时做了什么呢",
            "在做什么呢？"
          )
          select<Unit> {
            nextMessageAsync(subject).onAwait {
              val msg = it.message.contentToString().trimStart('在')
              data.activities[now - 1] = msg
              when {
                msg.contains(Regex("什么都没做|什么也没做|摸鱼")) -> {
                  data.activities[now - 1] = "摸鱼"
                  subject.sendMessageIn("啊这", "这样啊")
                  subject.sendMessage("那接下来得加油咯")
                }
                msg.contains(Regex("色色|打飞机|撸管|手冲")) -> {
                  subject.sendMessageIn("啧", "嘶")
                  subject.sendMessage("那不打扰你了")
                  subject.sendMessage("玩得愉快hhh")
                }
                msg.contains(Regex("坐车|开车")) -> {
                  subject.sendMessage("赶路很辛苦吧")
                  subject.sendMessage("祝你不虚此行")
                }
                msg.contains(Regex("午休|休息")) -> {
                  subject.sendMessage("休息得好吗？")
                  subject.sendMessage("恢复精力之后就继续工作吧")
                }
                msg.contains(Regex("[玩打](游戏|.游)|出去玩")) -> {
                  subject.sendMessage("那就……玩得开心")
                }
                else -> {
                  subject.sendMessageIn("不错嘛", "可以")
                  subject.sendMessageIn(
                    "休息一下再继续吧",
                    "继续之前，给自己一点放松时间吧"
                  )
                }
              }
            }
            onTimeout(300_000) {
              subject.sendMessage("还在忙吗……")
              subject.sendMessage("那不打扰你了")
              subject.sendMessage("不过，最好还是休息一下")
            }
          }
        }
        data.endOfDay -> {
          subject.sendMessageIn(
            "一天差不多结束了",
            "已经${data.endOfDay}点了"
          )
          subject.sendMessageIn(
            "回顾一下这一天吧……",
            "看看这一天你做了些什么……"
          )
          data.activities.keys.sorted().forEach { time ->
            subject.sendMessage("${time}点，你在${data.activities[time]}")
          }
          subject.sendMessage("${data.endOfDay}点，你在……")
          select<Unit> {
            nextMessageAsync(subject).onAwait {
              data.activities[data.endOfDay] =
                it.message.contentToString().trimStart('在')
              subject.sendMessage("……那么")
            }
            onTimeout(300_000) {
              subject.sendMessage("……算了")
            }
          }
          subject.sendMessageIn(
            "感觉怎么样？",
            "评价一下自己？"
          )
          select<Unit> {
            nextMessageAsync(subject).onAwait {
              data.assessment = it.message.contentToString()
              subject.sendMessageIn(
                "之后，差不多该休息了吧？",
                "到休息的时候了"
              )
            }
            onTimeout(300_000) {
              subject.sendMessage("不管怎样，现在到休息时间了")
            }
          }
          subject.sendMessageIn(
            "去洗漱吧",
            "关掉电脑吧"
          )
        }
      } } }
    }
  }

  fun onDisable() {
    timer.cancel()
  }
}
