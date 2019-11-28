package ch.zhdk.pose.model

import org.bytedeco.opencv.opencv_core.Point2d

class Marker(uniqueId: Int) : TrackingEntity(uniqueId) {
    var position = Point2d()

    // tracking relevant
    var matchedWithRegion = false

    // intensity detected by active region
    var intensity = 0.0
}