package ch.bildspur.util

import ch.fhnw.exakt.util.BatchingSequence
import processing.core.*
import processing.core.PConstants.QUAD_STRIP
import java.util.*
import kotlin.math.roundToInt

/**
 * Created by cansik on 04.02.17.
 */
fun Number.format(digits: Int) = java.lang.String.format("%.${digits}f", this)

fun Float.isApproximate(value: Double, error: Double): Boolean {
    return (Math.abs(Math.abs(this) - Math.abs(value)) < error)
}

fun Float.limit(min: Float, max: Float): Float {
    return Math.max(Math.min(max, this), min)
}

fun Float.map(start1 : Float, stop1 : Float, start2 : Float, stop2: Float) : Float {
    return start2 + (stop2 - start2) * ((this - start1) / (stop1 - start1))
}

fun Long.formatSeconds() : String {
    return (this / 1000.0).format(2)
}

fun Int.map(start1: Int, stop1: Int, start2: Int, stop2: Int): Int {
    return this.toDouble().map(
        start1.toDouble(),
        stop1.toDouble(),
        start2.toDouble(),
        stop2.toDouble()
    ).roundToInt()
}

fun Double.map(start1: Double, stop1: Double, start2: Double, stop2: Double): Double {
    return start2 + (stop2 - start2) * ((this - start1) / (stop1 - start1))
}

fun Double.isBetween(low: Double, high: Double): Boolean {
    return low < this && this < high
}

fun Double.isInMargin(limit: Double, margin: Double): Boolean {
    return this.isBetween(limit - margin, limit + margin)
}

fun Boolean.toFloat(): Float {
    return if (this) 1f else 0f
}

/**
 * Returns a single list of all elements from all arrays in the given array.
 */
fun <T> Array<out Array<out T>>.flatten(): List<T> {
    val result = ArrayList<T>(sumBy { it.size })
    for (element in this) {
        result.addAll(element)
    }
    return result
}

fun PGraphics.stackMatrix(block: (g: PGraphics) -> Unit) {
    this.pushMatrix()
    block(this)
    this.popMatrix()
}

fun PGraphics.draw(block: (g: PGraphics) -> Unit) {
    this.beginDraw()
    block(this)
    this.endDraw()
}

fun PGraphics.shape(block: (g: PGraphics) -> Unit) {
    this.beginShape()
    block(this)
    this.endShape(PApplet.CLOSE)
}

fun PGraphics.translate(vector: PVector) {
    this.translate(vector.x, vector.y, vector.z)
}

fun PGraphics.translateCenter() {
    this.translate(this.width / 2f, this.height / 2f)
}

fun PGraphics.rotate(vector: PVector) {
    this.rotateX(vector.x)
    this.rotateY(vector.y)
    this.rotateZ(vector.z)
}


fun Float.toRadians(): Float {
    return PApplet.radians(this)
}

fun Float.toDegrees(): Float {
    return PApplet.degrees(this)
}

fun PGraphics.createRod(r: Float, h: Float, detail: Int): PShape {
    textureMode(PApplet.NORMAL)
    val sh = createShape()
    sh.beginShape(QUAD_STRIP)
    for (i in 0..detail) {
        val angle = PApplet.TWO_PI / detail
        val x = Math.sin((i * angle).toDouble()).toFloat()
        val z = Math.cos((i * angle).toDouble()).toFloat()
        val u = i.toFloat() / detail
        sh.normal(x, 0f, z)
        sh.vertex(x * r, -h / 2, z * r, u, 0f)
        sh.vertex(x * r, +h / 2, z * r, u, 1f)
    }
    sh.endShape()
    return sh
}

fun PGraphics.cross(x: Float, y: Float, size: Float) {
    this.line(x, y - size, x, y + size)
    this.line(x - size, y, x + size, y)
}

fun PGraphics.cylinder(sides: Int, r1: Float, r2: Float, h: Float) {
    val angle = (360 / sides).toFloat()
    val halfHeight = h / 2
    // top
    this.beginShape()
    for (i in 0 until sides) {
        val x = PApplet.cos(PApplet.radians(i * angle)) * r1
        val y = PApplet.sin(PApplet.radians(i * angle)) * r1
        this.vertex(x, y, -halfHeight)
    }
    this.endShape(PApplet.CLOSE)
    // bottom
    this.beginShape()
    for (i in 0 until sides) {
        val x = PApplet.cos(PApplet.radians(i * angle)) * r2
        val y = PApplet.sin(PApplet.radians(i * angle)) * r2
        this.vertex(x, y, halfHeight)
    }
    this.endShape(PApplet.CLOSE)
    // draw body
    this.beginShape(PApplet.TRIANGLE_STRIP)
    for (i in 0 until sides + 1) {
        val x1 = PApplet.cos(PApplet.radians(i * angle)) * r1
        val y1 = PApplet.sin(PApplet.radians(i * angle)) * r1
        val x2 = PApplet.cos(PApplet.radians(i * angle)) * r2
        val y2 = PApplet.sin(PApplet.radians(i * angle)) * r2
        this.vertex(x1, y1, -halfHeight)
        this.vertex(x2, y2, halfHeight)
    }
    this.endShape(PApplet.CLOSE)
}

fun PGraphics.imageRect(image: PImage, x: Float, y: Float, width: Float, height: Float) {
    val ratio = if (width - image.width < height - image.height) width / image.width else height / image.height
    this.image(image, x, y, image.width * ratio, image.height * ratio)
}

fun <T> Sequence<T>.batch(n: Int): Sequence<List<T>> {
    return BatchingSequence(this, n)
}

fun Boolean.toSign(): Int {
    return if (this) 1 else -1
}

fun Boolean.toInvertSign(): Int {
    return if (this) -1 else 1
}

fun Boolean.toInt(): Int {
    return if (this) 1 else 0
}

fun PVector.flip(flipX: Boolean, flipY: Boolean, flipZ: Boolean) {
    this.x = this.x * flipX.toInvertSign()
    this.y = this.y * flipY.toInvertSign()
    this.z = this.z * flipZ.toInvertSign()
}