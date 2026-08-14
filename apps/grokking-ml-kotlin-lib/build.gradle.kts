plugins {
    kotlin("jvm") version "2.2.20"
}

group = "com.example.grokkingmllib"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    // Smile が原著の scikit-learn の役割を担う（回帰・分類・決定木・SVM・MLP）
    implementation("com.github.haifengl:smile-core:3.1.1")
    // Multik が NumPy の役割を担う（多次元配列と要素演算）
    implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
    // Kotlin DataFrame が pandas の役割を担う（CSV 読み込みと列操作）
    implementation("org.jetbrains.kotlinx:dataframe:0.15.0")
    // Kandy が matplotlib の役割を担う（ノートブック上の可視化）
    implementation("org.jetbrains.kotlinx:kandy-lets-plot:0.8.0")
    // 原著の第 12 章がそのまま XGBoost を使うので、JVM 版を入れる
    implementation("ml.dmlc:xgboost4j_2.12:2.0.3")

    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(21)
}

tasks.test {
    useJUnitPlatform()
    // #15 の MNIST は 6 万枚 × 784 画素を double で持つので 400 MB 近くになる。
    // 既定のヒープではテスト用 JVM が落ちる（java.io.EOFException として現れる）
    maxHeapSize = "4g"
    testLogging {
        events("passed", "failed", "skipped")
    }
}
