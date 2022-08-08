package one.paro.buromu

import kotlinx.coroutines.CoroutineScope

import net.mamoe.mirai.console.data.AutoSavePluginData
import net.mamoe.mirai.console.data.value
import net.mamoe.mirai.event.EventPriority
import net.mamoe.mirai.event.globalEventChannel
import net.mamoe.mirai.event.subscribeGroupMessages
import net.mamoe.mirai.message.data.MessageSource.Key.quote
import net.mamoe.mirai.message.data.PlainText
import net.mamoe.mirai.message.data.at

import one.paro.buromu.ChatState.Companion.filterState
import one.paro.buromu.Main.reload

object Jielong: CoroutineScope by Main {
  private object Data: AutoSavePluginData("Buromu.Jielong") {
    val longs by value<MutableMap<Long, MutableList<Long>>>()
  }

  fun onEnable() {
    Data.reload()
    globalEventChannel().filterState(null).subscribeGroupMessages(
      priority = EventPriority.NORMAL
    ) {
      (atBot() and contains("开新龙啦")) {
        Data.longs[group.id] = mutableListOf()
        group.sendMessage("新龙开始报名！At我说“我要报名”即可报名，At我说“我要龙头”即可将成为龙头！")
      }
      (atBot() and contains("我要报名")).invoke {
        val queue = Data.longs[group.id] ?: run {
          subject.sendMessage(message.quote() + "新龙还没开始呢")
          return@invoke
        }
        if (queue.size == 0) queue.add(sender.id)
        else {
          queue.remove(sender.id)
          queue.add((1..queue.size).random(), sender.id)
        }
        subject.sendMessage(sender.at() + "报名成功！")
      }
      (atBot() and contains("我要龙头")).invoke {
        val queue = Data.longs[group.id] ?: run {
          subject.sendMessage(message.quote() + "新龙还没开始呢")
          return@invoke
        }
        if (queue.size == 0) queue.add(sender.id)
        else {
          if (queue[0] == sender.id) {
            subject.sendMessage(sender.at() + "你已经是龙头了！")
            return@invoke
          }
          queue.remove(sender.id)
          queue.add((1..queue.size).random(), queue[0])
          queue[0] = sender.id
        }
        subject.sendMessage(sender.at() + "报名成功！")
      }
      (atBot() and contains("我要取消报名")).invoke {
        val queue = Data.longs[group.id] ?: run {
          subject.sendMessage(message.quote() + "新龙还没开始呢")
          return@invoke
        }
        queue.remove(sender.id)
        subject.sendMessage(sender.at() + "取消报名成功！")
      }
      (atBot() and contains("新龙报名结束")).invoke {
        val queue = Data.longs[group.id] ?: run {
          subject.sendMessage(message.quote() + "还没有创建新龙")
          return@invoke
        }
        subject.sendMessage("新龙报名结束啦！本次接龙的顺序如下：")
        subject.sendMessage(queue.mapIndexed { idx, it ->
          PlainText((idx + 1).toString() + ". ") +
            (group[it]?.at() ?: PlainText("（已退群）"))
        } .reduce { acc, it -> acc + "\n" + it })
        Data.longs.remove(group.id)
      }
    }
  }
}
