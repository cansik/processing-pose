package ch.zhdk.pose.pipeline

import ch.zhdk.pose.config.PipelineConfig
import ch.zhdk.pose.io.InputProvider
import ch.zhdk.pose.javacv.drawCircle
import ch.zhdk.pose.javacv.height
import ch.zhdk.pose.javacv.toPoint
import ch.zhdk.pose.javacv.width
import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.opencv.global.opencv_core.CV_32F
import org.bytedeco.opencv.global.opencv_core.minMaxLoc
import org.bytedeco.opencv.global.opencv_dnn.*
import org.bytedeco.opencv.global.opencv_imgcodecs.imwrite
import org.bytedeco.opencv.opencv_core.*
import java.nio.file.Paths
import kotlin.math.roundToInt


class LightOpenPoseTorchPipeline(config: PipelineConfig, inputProvider: InputProvider, pipelineLock: Any = Any()) :
    Pipeline(config, inputProvider, pipelineLock) {

    private val weightPath = Paths.get("models/light/checkpoint_iter_370000.pth").toAbsolutePath()

    private val net = readNetFromTorch(weightPath.toString())

    init {
        //net.setPreferableBackend(DNN_BACKEND_VKCOM)
        //net.setPreferableTarget(DNN_TARGET_OPENCL)
    }

    override fun detectPose(frame: Mat, timestamp: Long) {
        // prepare input image
        val frameHeight = 368
        val frameWidth = (frameHeight.toFloat() / frame.height() * frame.width()).roundToInt()

        val zeroScalar = Scalar(0.0, 0.0, 0.0, 0.0)
        val inpBlob = blobFromImage(frame, 1.0 / 255, Size(frameWidth, frameHeight), zeroScalar, false, false, CV_32F)
        net.setInput(inpBlob)

        val output = net.forward()

        val nPoints = output.size(1)
        val matHeight = output.size(2)
        val matWidth = output.size(3)

        println("nPoints: $nPoints")

        // find the position of the body parts
        val points = mutableListOf<Point2f>()
        for(i in 0 until nPoints) {
            val probMap = Mat(matHeight, matWidth, CV_32F, output.ptr(0, i))

            // store prob map
            imwrite("maps/probMap_$i.png", probMap)

            val p = Point2f(-1f,-1f)

            val minVal = DoublePointer()
            val minLoc = Point()
            val confidence = DoublePointer()
            val maxLoc = Point()

            minMaxLoc(probMap, minVal, confidence, minLoc, maxLoc, null)

            if(confidence.isNull)
                continue

            if (confidence.get() > config.threshold.value)
            {
                p.x(maxLoc.x() + (frameWidth.toFloat() / matWidth))
                p.y(maxLoc.y() + (frameHeight.toFloat() / matHeight))

                frame.drawCircle(p.toPoint(), 10, AbstractScalar.MAGENTA)
            }
            points.add(p)
        }

        println("found ${points.size} points!")
    }
}