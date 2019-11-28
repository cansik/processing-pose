package ch.zhdk.pose.pipeline

import ch.bildspur.model.math.Float2
import ch.zhdk.pose.model.ActiveRegion
import ch.zhdk.pose.model.Marker
import org.bytedeco.opencv.opencv_core.Point
import kotlin.math.roundToInt

fun Float2.toPoint() : Point {
    return Point(this.x.roundToInt(), this.y.roundToInt())
}