package one.paro.buromu

import java.io.File
import java.io.IOException
import java.io.InputStreamReader
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

import net.mamoe.mirai.console.data.AutoSavePluginConfig
import net.mamoe.mirai.console.data.value
import net.mamoe.mirai.event.events.FriendMessageEvent
import net.mamoe.mirai.event.globalEventChannel

import one.paro.buromu.Main.reload

object ServerStatus: CoroutineScope by Main {
  private val logger = Main.logger.derive(this::class)

  private object Config: AutoSavePluginConfig("Buromu.ServerStatus") {
    val path by value<String>()
    val users by value<Set<Long>>()
  }

  private suspend fun queryServerStatus(): Boolean {
    return try {
      val output = withContext(Dispatchers.IO) {
        val proc = Runtime.getRuntime()
          .exec(arrayOf("/usr/bin/screen", "-ls"))
        InputStreamReader(proc.inputStream).readLines().joinToString("\n")
      }
      output.contains("minecraft")
    } catch (e: IOException) {
      logger.error("Failed to query server status", e)
      throw e
    }
  }

  fun onEnable() {
    Config.reload()
    globalEventChannel().subscribeAlways<FriendMessageEvent> {
      if (!Config.users.contains(it.subject.id)) return@subscribeAlways
      when (message.contentToString()) {
        "服务器开着吗" -> {
          try {
            if (queryServerStatus()) subject.sendMessage("开着呢")
            else subject.sendMessage("没开")
          } catch (e: IOException) {
            subject.sendMessage("出错了！")
          }
        }
        "启动服务器" -> {
          try {
            if (queryServerStatus()) subject.sendMessage("服务器已经开啦！")
            else try {
              withContext(Dispatchers.IO) {
                Runtime.getRuntime().exec(
                  arrayOf("/usr/bin/screen", "-dmS", "minecraft", "java",
                    "-Xmx2G", "-jar", "fabric-server-launch.jar", "nogui"),
                  null, File(Config.path)
                ).onExit().join()
              }
              subject.sendMessage("完成，稍等一会就能上服")
            } catch (e: IOException) {
              logger.error("Failed to launch server", e)
            }
          } catch (e: IOException) {
            subject.sendMessage("出错了！")
          }
        }
        "关闭服务器" -> {
          try {
            if (!queryServerStatus()) subject.sendMessage("服务器还没开呢！")
            else try {
              withContext(Dispatchers.IO) {
                Runtime.getRuntime().exec(arrayOf(
                  "/usr/bin/screen", "-S", "minecraft",
                  "-X", "stuff", "stop\n"
                )).onExit().join()
              }
              subject.sendMessage("完成，服务器很快就会关闭")
            } catch (e: IOException) {
              logger.error("Failed to stop server", e)
            }
          } catch (e: IOException) {
            subject.sendMessage("出错了！")
          }
        }
      }
    }
  }
}
