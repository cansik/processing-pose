package ch.zhdk.pose.pipeline

import ch.zhdk.pose.config.PipelineConfig
import ch.zhdk.pose.io.InputProvider
import org.bytedeco.opencv.opencv_core.Mat

class PassthroughPipeline(config: PipelineConfig, inputProvider: InputProvider, pipelineLock: Any = Any()) :
    Pipeline(config, inputProvider, pipelineLock) {

    override fun detectPose(frame: Mat, timestamp: Long) {
    }
}