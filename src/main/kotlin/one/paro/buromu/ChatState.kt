package one.paro.buromu

import kotlin.reflect.KClass
import kotlin.reflect.full.isSubclassOf
import kotlinx.coroutines.CoroutineScope

import net.mamoe.mirai.contact.Contact
import net.mamoe.mirai.event.EventChannel
import net.mamoe.mirai.event.events.MessageEvent

interface ChatState {
  companion object: CoroutineScope by Main {
    private val states = mutableMapOf<Contact, KClass<out ChatState>>()

    fun Contact.getState() = states[this]

    fun Contact.switch(state: KClass<out ChatState>?) {
      if (state == null) states.remove(this)
      else states[this] = state
    }
    fun Contact.switch(state: ChatState?) {
      if (state == null) states.remove(this)
      else states[this] = state::class
    }

    inline fun <T> Contact.withState(state: ChatState?, block: () -> T): T {
      val prevState = this.getState()
      this.switch(state)
      val result = block()
      this.switch(prevState)
      return result
    }

    fun <E : MessageEvent> EventChannel<E>.filterState(state: ChatState?) =
      if (state == null) this.filter { !states.containsKey(it.subject) }
      else this.filter {
        states[it.subject]?.isSubclassOf(state::class) ?: false
      }
    @JvmName("filterStateAuto")
    fun EventChannel<*>.filterState(state: ChatState?) =
      this.filterIsInstance<MessageEvent>().filterState(state)
  }
}
