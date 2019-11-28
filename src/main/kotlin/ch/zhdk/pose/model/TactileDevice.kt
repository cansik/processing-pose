package ch.zhdk.pose.model

import ch.zhdk.pose.javacv.angleOfInDeg
import ch.zhdk.pose.javacv.center
import ch.zhdk.pose.javacv.distance
import org.bytedeco.opencv.opencv_core.Point2d


class TactileDevice(uniqueId: Int) : TrackingEntity(uniqueId) {
    var position = Point2d()

    var normalizedPosition = Point2d()
    var normalizedIntensity: Double = 0.0
    var calibratedPosition = Point2d()

    var identifier = -1

    var rotation = 0.0

    var intensity = 0.0

    var markers = ArrayList<Marker>(3)

    var leftOriginMarker : Marker? = null
    var rightOriginMarker : Marker? = null
    var communicationMarker : Marker? = null
    var stopBitMarker : Marker? = null

    var matchedWithCentroid = false

    fun update() {
        if(markers.size < 2)
            return

        identifyMarkers()
        orderOriginMarker()

        updatePosition()

        if (markers.size < 3)
            return

        updateRotation()
    }

    private fun identifyMarkers() {
        if(markers.size < 3) {
            leftOriginMarker = markers[0]
            rightOriginMarker = markers[1]
            return
        }

        // analysis for 3 or more markers
        data class MarkerDistances(val m1 : Marker, val m2 : Marker, val distance : Double)

        // calculate distances between each points
        val distances = markers.take(markers.size - 1)
            .mapIndexed { i, marker ->
            markers.drop(i + 1).map {
                MarkerDistances(marker, it, marker.position.distance(it.position))
            }
        }.flatten()

        // get largest tuple for origin points
        val sortedDistances = distances.sortedByDescending { it.distance }
        val largest = sortedDistances.first()
        leftOriginMarker = largest.m1
        rightOriginMarker = largest.m2

        // find communication marker
        communicationMarker = markers.subtract(listOf(leftOriginMarker, rightOriginMarker)).first()
    }

    private fun orderOriginMarker() {
        // todo: implement decision of left & right
    }

    private fun updatePosition() {
        position = listOf(leftOriginMarker!!.position, rightOriginMarker!!.position).center()
    }

    private fun updateRotation() {
        this.rotation = position.angleOfInDeg(communicationMarker!!.position)
    }
}