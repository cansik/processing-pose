package ch.zhdk.pose.io

import ch.zhdk.pose.config.InputConfig
import org.bytedeco.javacv.Frame
import org.bytedeco.javacv.RealSense2FrameGrabber

class RealSense2InputProvider(
    val deviceNumber: Int = 0,
    width: Int = 640,
    height: Int = 480,
    val frameRate: Int = 30,
    var config: InputConfig
) : InputProvider(width, height) {

    private lateinit var rs2: RealSense2FrameGrabber

    override fun open() {
        rs2 = RealSense2FrameGrabber(deviceNumber)
        rs2.enableColorStream(width, height, frameRate)

        // open device
        rs2.open()

        rs2.start()
        super.open()
    }

    override fun read(): Frame {
        rs2.trigger()
        return rs2.grabColor()
    }

    override fun close() {
        rs2.stop()
        rs2.release()
        super.close()
    }

}