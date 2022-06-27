plugins {
    id("buromu.build")
    application
}

dependencies {
    implementation(project(":core"))
}

application {
    mainClass.set("buromu.app.AppKt")
}
