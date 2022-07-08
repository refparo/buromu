plugins {
  val kotlinVersion = "1.7.10"
  kotlin("jvm") version kotlinVersion
  kotlin("plugin.serialization") version kotlinVersion

  val miraiVersion = "2.12.0"
  id("net.mamoe.mirai-console") version miraiVersion
}

group = "one.paro"
version = "0.0.1"

repositories {
  mavenLocal()
  maven("https://mirrors.cloud.tencent.com/nexus/repository/maven-public/")
  mavenCentral()
}

dependencies {}

testing {}
