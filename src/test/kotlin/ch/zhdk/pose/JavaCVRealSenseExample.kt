package ch.zhdk.pose

import org.bytedeco.javacv.*


object JavaCVRealSenseExample {
    @JvmStatic
    fun main(args: Array<String>) {
        val grabber = RealSense2FrameGrabber(0)
        println("grabber created")

        grabber.enableIRStream(640, 480, 30)
        grabber.start()
        println("grabber started")

        var grabbedImage: Frame = grabber.grab()

        val canvasFrame = CanvasFrame("RealSense")
        canvasFrame.setCanvasSize(grabbedImage.imageWidth, grabbedImage.imageHeight)

        println("framerate = " + grabber.frameRate)

        while (canvasFrame.isVisible) {
            grabbedImage = grabber.grabIR()
            canvasFrame.showImage(grabbedImage)
        }

        grabber.stop()
        canvasFrame.dispose()
    }
}