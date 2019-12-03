package ch.zhdk.pose.pipeline

import ch.zhdk.pose.config.PipelineConfig
import ch.zhdk.pose.io.InputProvider
import ch.zhdk.pose.javacv.*
import org.bytedeco.javacpp.indexer.FloatBufferIndexer
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.opencv.global.opencv_core.CV_32F
import org.bytedeco.opencv.global.opencv_core.CV_8U
import org.bytedeco.opencv.global.opencv_dnn.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_core.*
import org.bytedeco.opencv.opencv_dnn.Net
import java.nio.file.Paths
import kotlin.math.roundToInt


class LightOpenPoseIEPipeline(config: PipelineConfig, inputProvider: InputProvider, pipelineLock: Any = Any()) :
    Pipeline(config, inputProvider, pipelineLock) {

    // https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/multi-person-openpose.cpp

    private val xmlPath = Paths.get("models/light/INT8/human-pose-estimation-0001.xml").toAbsolutePath()
    private val weightPath = Paths.get("models/light/INT8/human-pose-estimation-0001.bin").toAbsolutePath()

    private val net : Net
    private val nPoints = 18

    private data class KeyPoint(val id : Int, val location : Point2d, val probability : Float)

    init {
        // load native tiny tbb as a workaround
        System.load("/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_mac/lib/libmkl_tiny_tbb.dylib")
        println("libmkl_tiny_tbb.dylib loaded!")

        // load network
        net = readNetFromModelOptimizer(xmlPath.toString(), weightPath.toString())
        net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(DNN_TARGET_CPU)
    }

    override fun detectPose(frame: Mat, timestamp: Long) {
        // prepare input image
        val frameHeight = 368
        val frameWidth = (frameHeight.toFloat() / frame.height() * frame.width()).roundToInt()
        val size = Size(frameWidth, frameHeight)

        val zeroScalar = Scalar(0.0, 0.0, 0.0, 0.0)
        val inpBlob  = blobFromImage(frame, 1.0, size, zeroScalar, false, false, CV_32F)
        net.setInput(inpBlob)

        val output = net.forward()
        val netOutputParts = splitNetOutputBlobToParts(frame.size(), output)

        (0 until nPoints).map { i ->
            val keyPoints = extractKeyPoints(netOutputParts[i], config.threshold.value, i)

            // mark keypoints
            keyPoints.forEach {
                frame.drawCircle(it.location.toPoint(), 5, AbstractScalar.RED)
            }
        }
    }

    private fun splitNetOutputBlobToParts(targetSize : Size, output : Mat) : List<Mat> {
        val nParts = output.size(1)
        val matHeight = output.size(2)
        val matWidth = output.size(3)

        return (0 until nParts).map {i ->
            val probMap = Mat(matHeight, matWidth, CV_32F, output.ptr(0, i))

            val dst = Mat()
            resize(probMap, dst, targetSize)
            dst
        }
    }

    private fun extractKeyPoints(probMap : Mat, threshold : Double, index : Int) : List<KeyPoint> {
        // smooth prob map
        val smoothProbMap = Mat()
        GaussianBlur(probMap, smoothProbMap, Size( 3, 3 ), 0.0,  0.0, 0)

        smoothProbMap.threshold(threshold, 255.0, THRESH_BINARY)
        smoothProbMap.convertTo(smoothProbMap, CV_8U)

        imwrite("maps/k_$index.bmp", smoothProbMap)

        // create indexer
        val indexer = probMap.createIndexer<FloatRawIndexer>()

        // extract points
        val components = smoothProbMap.connectedComponentsWithStats().getConnectedComponents()
        return components.map {
            val pos = it.centroid.toPoint()
            KeyPoint(index, it.centroid, indexer.get(pos.y().toLong(), pos.x().toLong()))
        }
    }
}