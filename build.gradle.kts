plugins {
  val kotlinVersion = "1.7.10"
  kotlin("jvm") version kotlinVersion
  kotlin("plugin.serialization") version kotlinVersion

  val miraiVersion = "2.11.1"
  id("net.mamoe.mirai-console") version miraiVersion
}

repositories {
  mavenCentral()
}

dependencies {
}

testing {
}
