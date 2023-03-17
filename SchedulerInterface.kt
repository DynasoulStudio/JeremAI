package com.dynasoulstudio.jeremai

import android.graphics.Bitmap
import android.widget.ImageView
import kotlin.math.ceil

interface SchedulerInterface {
    val timesteps: IntArray
    val initNoiseSigma: Float
    fun setTimesteps(numInferenceSteps: Int)
    fun scaleModelInput(sample: Array<Array<Array<FloatArray>>>, timestep: Int): Array<Array<Array<FloatArray>>>
    fun step(modelOutput: Array<Array<Array<FloatArray>>>, timestep: Int, sample: Array<Array<Array<FloatArray>>>, eta: Float = 0.0f, useClippledModelOutput: Boolean = false, seed: Long? = null, varianceNoise: Array<Array<Array<FloatArray>>>? = null, returnDict: Boolean = true, imageCanvas: ImageView? = null): Any
    fun linearSpace(start: Float, stop: Float, num: Int): FloatArray {
        val step = (stop - start) / (num.toFloat() - 1)
        return ((0 until num).map { start + it * step }).toFloatArray()
    }

    fun linearSpace(start: Int, stop: Int, num: Int): IntArray {
        val step = (stop - start) / (num - 1)
        return ((0 until num).map { start + it * step }).toIntArray()
    }

    fun linearSpace(start: Double, stop: Double, num: Int): DoubleArray {
        val step = (stop - start) / (num - 1.0)
        return ((0 until num).map { start + it * step }).toDoubleArray()
    }

    fun arange(start: Int, stop: Int, step: Int = 1): IntArray {
        val size = ceil((stop.toDouble() - start.toDouble()) / step.toDouble()).toInt()
        val out = IntArray(size) { start + it * step }
        return out
    }

    fun arange(start: Double, stop: Double, step: Double = 1.0): DoubleArray {
        val size = ceil((stop - start) / step).toInt()
        val out = DoubleArray(size) { start + it * step }
        return out
    }

    fun arange(start: Float, stop: Float, step: Float = 1.0f): FloatArray {
        val size = ceil((stop - start) / step).toInt()
        val out = FloatArray(size) { start + it * step }
        return out
    }

    fun cumProd(input: FloatArray): FloatArray {
        val result = FloatArray(input.size)
        result[0] = input[0]
        for (i in 1 until input.size) {
            result[i] = result[i - 1] * input[i]
        }
        return result
    }

    fun cumProd(input: DoubleArray): DoubleArray {
        val result = DoubleArray(input.size)
        result[0] = input[0]
        for (i in 1 until input.size) {
            result[i] = result[i - 1] * input[i]
        }
        return result
    }

    fun interp(x: DoubleArray, xp: DoubleArray, fp: DoubleArray): DoubleArray {
        val result = DoubleArray(x.size)
        for (i in x.indices) {
            val xVal = x[i]
            var index = -1
            for (j in 0 until xp.size - 1) {
                if (xVal in xp[j]..xp[j + 1]) {
                    index = j
                    break
                }
            }
            if (index == -1) {
                result[i] = if (xVal <= xp.first()) fp.first() else fp.last()
            } else {
                val x1 = xp[index]
                val x2 = xp[index + 1]
                val y1 = fp[index]
                val y2 = fp[index + 1]
                result[i] = y1 + (y2 - y1) * (xVal - x1) / (x2 - x1)
            }
        }
        return result
    }

    fun interp(x: FloatArray, xp: FloatArray, fp: FloatArray): FloatArray {
        val result = FloatArray(x.size)
        for (i in x.indices) {
            val xVal = x[i]
            var index = -1
            for (j in 0 until xp.size - 1) {
                if (xVal in xp[j]..xp[j + 1]) {
                    index = j
                    break
                }
            }
            if (index == -1) {
                result[i] = if (xVal <= xp.first()) fp.first() else fp.last()
            } else {
                val x1 = xp[index]
                val x2 = xp[index + 1]
                val y1 = fp[index]
                val y2 = fp[index + 1]
                result[i] = y1 + (y2 - y1) * (xVal - x1) / (x2 - x1)
            }
        }
        return result
    }

    fun concatenate(array: FloatArray, value: Float): FloatArray {
        return array + value
    }

    fun concatenate(array: DoubleArray, value: Double): DoubleArray {
        return array + value
    }

    fun concatenate(array: FloatArray, array2: FloatArray): FloatArray {
        return array + array2
    }

    fun nonzero(array: FloatArray): Int {
        for (i in array.indices) {
            if (array[i] != 0.0f) {
                return i
            }
        }
        return -1
    }

    fun nonzero(array: DoubleArray): Int {
        for (i in array.indices) {
            if (array[i] != 0.0) {
                return i
            }
        }
        return -1
    }

    fun sigmoid(x: DoubleArray): DoubleArray {
        return x.map { 1f / (1f + Math.exp(-it.toDouble())) }.toDoubleArray()
    }

    fun IntArray.repeat(n: Int): IntArray {
        return IntArray(size * n) { this[it % size] }
    }

    fun FloatArray.repeat(n: Int): FloatArray {
        return FloatArray(size * n) { this[it % size] }
    }

    fun List<Float>.repeat(n: Int): List<Float>{
        return List<Float>(size * n) { this[it % size] }
    }

    fun updateMainImageView(imageCanvas: ImageView, image: Bitmap) {
        imageCanvas.setImageBitmap(image)
    }
}