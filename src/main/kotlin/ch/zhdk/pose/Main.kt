package ch.zhdk.pose

import ch.bildspur.configuration.ConfigurationController
import ch.zhdk.pose.config.AppConfig
import javafx.application.Platform
import javafx.stage.Stage
import org.bytedeco.javacpp.Loader
import org.bytedeco.opencv.global.opencv_core


class Main {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            Main().startApplication(args)
        }
    }

    fun startApplication(args: Array<String>) {
        val configuration = ConfigurationController("Pose Estimation", "zhdk", "pose")
        var appConfig = configuration.loadAppConfig()

        // initialize opencv (needed for realsense camera)
        Loader.load(opencv_core::class.java)

        // use a fresh config while debugging
        if (args.contains("-dev"))
            appConfig = AppConfig()

        // start configuration app
        Platform.startup {
            val window = ConfigurationWindow(configuration, appConfig)
            val stage = Stage()
            window.start(stage)
        }

        // start main app
        TrackingApplication.start(appConfig)
    }
}