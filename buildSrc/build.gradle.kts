plugins {
  `kotlin-dsl`
}

repositories {
  gradlePluginPortal()
}

dependencies {
  val kotlinVersion = "1.7.0"
  implementation("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion")
  implementation("org.jetbrains.kotlin:kotlin-serialization:$kotlinVersion")

  val miraiVersion = "2.11.1"
  implementation("net.mamoe:mirai-console-gradle:$miraiVersion")
}
