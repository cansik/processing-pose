package ch.zhdk.pose

import ch.bildspur.util.TimeKeeper

object TimingTest {
    @JvmStatic
    fun main(args: Array<String>) {
        println("Timing Test")

        (0 until 100).forEach {
            println("MILLI: ${TimeKeeper.millis()} ")
        }
    }
}