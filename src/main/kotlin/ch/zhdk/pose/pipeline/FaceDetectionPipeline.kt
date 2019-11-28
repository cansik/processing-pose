package ch.zhdk.pose.pipeline

import ch.zhdk.pose.config.PipelineConfig
import ch.zhdk.pose.io.InputProvider
import ch.zhdk.pose.javacv.drawRect
import ch.zhdk.pose.javacv.height
import ch.zhdk.pose.javacv.width
import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.opencv.global.opencv_core.CV_32F
import org.bytedeco.opencv.global.opencv_dnn.*
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.opencv_core.*
import java.nio.file.Paths
import kotlin.math.roundToInt

class FaceDetectionPipeline(config: PipelineConfig, inputProvider: InputProvider, pipelineLock: Any = Any()) :
    Pipeline(config, inputProvider, pipelineLock) {

    private val protoPath = Paths.get("models/face/deploy.prototxt.txt").toAbsolutePath()
    private val weightPath = Paths.get("models/face/res10_300x300_ssd_iter_140000.caffemodel").toAbsolutePath()

    private val net = readNetFromCaffe(protoPath.toString(), weightPath.toString())

    init {
        net.setPreferableTarget(DNN_TARGET_OPENCL)
    }

    override fun detectPose(frame: Mat, timestamp: Long) {

        //resize the image to match the input size of the model
        val inputSize = Size((frame.width() * config.imageScale.value).roundToInt(), (frame.height() * config.imageScale.value).roundToInt())
        val inpBlob = blobFromImage(frame, 1.0, inputSize, Scalar(104.0, 177.0, 123.0, 0.0), false, false, CV_32F)
        net.setInput(inpBlob)

        val output = net.forward()

        //extract a 2d matrix for 4d output matrix with form of (number of detections x 7)
        val ne = Mat(Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0))

        // create indexer to access elements of the matric
        val srcIndexer: FloatIndexer = ne.createIndexer()

        //iterate to extract elements
        for (i in 0 until output.size(3)) {
            val confidence = srcIndexer[i.toLong(), 2]
            val f1 = srcIndexer[i.toLong(), 3]
            val f2 = srcIndexer[i.toLong(), 4]
            val f3 = srcIndexer[i.toLong(), 5]
            val f4 = srcIndexer[i.toLong(), 6]

            if (confidence > config.threshold.value) {
                val tx = f1 * frame.width() //top left point's x
                val ty = f2 * frame.height() //top left point's y
                val bx = f3 * frame.width() //bottom right point's x
                val by = f4 * frame.height() //bottom right point's y

                frame.drawRect(
                    Rect(Point(tx.toInt(), ty.toInt()), Point(bx.toInt(), by.toInt())),
                    AbstractScalar.MAGENTA,
                    1
                )
            }
        }
    }
}