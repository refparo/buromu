package one.paro.buromu

import java.util.*
import java.time.DateTimeException
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
    var noDisturb = false
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
    val noDisturb: Boolean,
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
          Data[subject] = UserData()
          subject.sendMessageIn("好啊", "好")
          subject.sendMessageIn(
            "不过，既然要求了，就要听话哦",
            "但可不要反悔啊"
          )
        }
      }
      finding(Regex("(不用|不要|别)帮我管理时间")) {
        if (Data.contains(subject)) {
          Data.remove(subject)
          subject.sendMessage("好吧")
          subject.sendMessageIn(
            "希望你没有我帮忙也能管好自己的时间",
            "你能自己管理时间那是最好"
          )
        } else {
          subject.sendMessage("我怎么不记得你有叫我帮过你")
        }
      }
      finding(Regex("""我的工作时间是(\d{1,2})点到(\d{1,2})点""")) {
        val data = Data[subject] ?: return@finding
        val start = it.groupValues[1].toInt()
        val end = it.groupValues[2].toInt()
        if (!(0..23).contains(start) || !(0..23).contains(end)) {
          subject.sendMessage("你的工作时间……是地球时间吗？")
        } else when {
          start == end -> {
            Data.remove(subject)
            subject.sendMessage("你的意思是，你不用工作？")
            subject.sendMessage("那我以后就不帮你管理时间了")
          }
          start > end -> {
            subject.sendMessage("不建议熬夜工作噢")
          }
          else -> {
            data.startOfDay = start
            data.endOfDay = end
            subject.sendMessage("了解了")
            subject.sendMessage("我会按这个时间提醒你的")
          }
        }
      }
      suspend fun recordPastActivity(
        subject: Friend, time: Int, activity: String, now: Int
      ) {
        val data = Data[subject] ?: return
        val validRange = data.startOfDay until data.endOfDay
        if (!validRange.contains(now)) {
          subject.sendMessage("工作时间已经过了，现在告诉我我也不会记录噢")
        } else if (!validRange.contains(time)) {
          subject.sendMessage("工作时间外的活动也要记录吗？")
        } else when {
          time < now -> {
            data.activities[time] = activity
            subject.sendMessageIn("知道了", "收到")
          }
          time == now -> {
            subject.sendMessage("${time}点还没结束呢！")
          }
          else -> {
            subject.sendMessage("${time}点还没到呢！")
          }
        }
      }
      matching(Regex("""(\d{1,2})点我在(.+)""")) {
        val (time, activity) = it.destructured
        val now = LocalTime.now().hour
        recordPastActivity(subject, time.toInt(), activity, now)
      }
      matching(Regex("刚才我?在(.+)")) {
        val (activity) = it.destructured
        val now = LocalTime.now().hour
        recordPastActivity(subject, now - 1, activity, now)
      }
      suspend fun taskToday(
        subject: Friend, length: Int, task: String, noDisturb: Boolean
      ) {
        val data = Data[subject] ?: return
        val now = LocalTime.now().hour
        if (now < data.startOfDay) {
          subject.sendMessage("工作时间还没到呢")
          subject.sendMessage("等到了工作时间再说吧")
          return
        }
        data.tasks.addAll(task.split(',', '，', '、').filter { it.isNotEmpty() })
        if (noDisturb) {
          data.noDisturb = true
          if (now + length >= data.endOfDay) {
            subject.sendMessage("要那么长时间吗")
            subject.sendMessage("那我今天都不打扰你了")
          } else {
            launch {
              delay(length.hours.inWholeMilliseconds)
              data.noDisturb = false
            }
            subject.sendMessage("明白了")
          }
        } else {
          subject.sendMessage("明白了，我会提醒你的")
        }
      }
      matching(Regex("等会我?[要会](.+?)([ ,，]*别打扰我)?")) {
        val (task, noDisturb) = it.destructured
        taskToday(subject, 1, task, noDisturb != "")
      }
      matching(Regex("""之后(\d{1,2})小时我?[要会](.+?)([ ,，]*别打扰我)?""")) {
        val (length, task, noDisturb) = it.destructured
        taskToday(subject, length.toInt(), task, noDisturb != "")
      }
      suspend fun taskFuture(
        subject: Friend, date: LocalDate, task: String, noDisturb: Boolean
      ) {
        val data = Data[subject] ?: return
        if (date.isBefore(LocalDate.now())) {
          subject.sendMessage("但是那一天已经过去了啊……你不会穿越了吧？")
          return
        }
        data.calendar[date.toString()] = UserDay(
          noDisturb,
          (data.calendar[date.toString()]?.tasks ?: setOf()) +
            task.split(',', '，', '、').filter { it.isNotEmpty() },
          mapOf(),
          null
        )
        subject.sendMessage("明白了，我会提醒你的")
      }
      matching(Regex("([今明后])天我?(整天都)?[要会](.+)")) {
        val (relDate, noDisturb, task) = it.destructured
        val date = LocalDate.now().plusDays(when (relDate) {
          "今" -> 0
          "明" -> 1
          "后" -> 2
          else -> throw IllegalArgumentException(
            "relDate == $relDate, which should be impossibe")
        })
        taskFuture(subject, date, task, noDisturb != "")
      }
      matching(Regex("""(\d{1,2})号我?(整天都)?[要会](.+)""")) {
        val (dayOfMonth, noDisturb, task) = it.destructured
        val date = try {
          LocalDate.now().withDayOfMonth(dayOfMonth.toInt())
        } catch (e: DateTimeException) {
          if (Data.contains(subject))
            subject.sendMessage("这日期看起来不像是地球历的啊")
          return@matching
        }
        taskFuture(subject, date, task, noDisturb != "")
      }
      matching(Regex("""(\d{1,2})月(\d{1,2})号我?(整天都)?[要会](.+)""")) {
        val (month, dayOfMonth, noDisturb, task) = it.destructured
        val date = try {
          LocalDate.now()
            .withMonth(month.toInt())
            .withDayOfMonth(dayOfMonth.toInt())
        } catch (e: DateTimeException) {
          if (Data.contains(subject))
            subject.sendMessage("这日期看起来不像是地球历的啊")
          return@matching
        }
        taskFuture(subject, date, task, noDisturb != "")
      }
      matching(Regex("我已经完成了(.+)")) {
        val data = Data[subject] ?: return@matching
        val (doneTasks) = it.destructured
        data.tasks
          .filter { doneTasks.contains(it) }
          .forEach { data.tasks.remove(it) }
        subject.sendMessage("了解了")
        if (data.tasks.isNotEmpty())
          subject.sendMessage("这样的话，你还要${data.tasks.joinToString("、")}")
        else
          subject.sendMessage("这样今天的目标就全都完成啦")
      }
    }
  }

  private val timer = Timer()
  private val task = object: TimerTask() {
    override fun run() {
      val now = LocalTime.now().hour
      Data.forEach { (subject, data) -> launch { when (now) {
        data.startOfDay -> {
          data.calendar[LocalDate.now().toString()]?.let {
            data.noDisturb = it.noDisturb
            data.tasks.addAll(it.tasks)
          }
          subject.sendMessageIn(
            "该干活啦！",
            "工作时间到了！"
          )
          if (data.needWakeUp) {
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
          if (data.tasks.isNotEmpty())
            subject.sendMessage("别忘了今天要${data.tasks.joinToString("、")}")
        }
        in data.startOfDay until data.endOfDay -> {
          if (data.noDisturb) return@launch
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
              val doneTasks = data.tasks.filter { msg.contains(it) }
                .onEach { data.tasks.remove(it) }
              if (doneTasks.isNotEmpty()) {
                subject.sendMessage("不错嘛，完成了${doneTasks.size}项目标")
                subject.sendMessageIn(
                  "休息一下再继续吧",
                  "继续之前，给自己一点放松时间吧"
                )
                if (data.tasks.isNotEmpty())
                  subject.sendMessage("接下来还要${data.tasks.joinToString("、")}")
                else
                  subject.sendMessage("这样今天的目标就全都完成啦")
              } else when {
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
          subject.sendMessage("${data.endOfDay - 1}点，你在……")
          select<Unit> {
            nextMessageAsync(subject).onAwait {
              data.activities[data.endOfDay - 1] =
                it.message.contentToString().trimStart('在')
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
          data.calendar[LocalDate.now().toString()] =
            UserDay(
              false,
              data.tasks.toSet(),
              data.activities.toMap(),
              data.assessment
            )
          data.tasks.clear()
          data.activities.clear()
          data.assessment = null
        }
      } } }
    }
  }

  fun onDisable() {
    timer.cancel()
  }
}
