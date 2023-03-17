package com.dynasoulstudio.jeremai

import android.graphics.Bitmap
import android.view.View
import android.widget.ImageView
import java.util.Random
import kotlin.math.ceil
import kotlin.math.pow
import kotlin.math.sqrt

data class EulerAncestralDiscreteSchedulerOutput(val prevSample: Array<Array<Array<FloatArray>>>, val predOriginalSample: Array<Array<Array<FloatArray>>>)

internal class EulerScheduler(
    val numTrainTimesteps: Int = 1000,
    val betaStart: Float = 0.0001f,
    val betaEnd: Float = 0.02f,
    val betaSchedule: String = "linear",
    val trainedBetas: FloatArray? = null,
    val predictionType: String = "epsilon"
): SchedulerInterface {
    var betas: FloatArray
    var alphas: FloatArray
    var alphasCumprod: FloatArray
    var sigmas: FloatArray
    override var initNoiseSigma: Float = 0f
    var numInferenceSteps: Int? = null
    var isScaleInputCalled: Boolean
    override lateinit var timesteps: IntArray


    init {
        if (trainedBetas != null) {
            betas = trainedBetas
        } else if (betaSchedule == "linear") {
            betas = linearSpace(betaStart,betaEnd,numTrainTimesteps)
        } else if (betaSchedule == "scaled_linear") {
            betas = linearSpace(sqrt(betaStart),sqrt(betaEnd),numTrainTimesteps)
            //betas = betas.map {it.pow(2.0f)}.toFloatArray()
            betas.forEachIndexed { i, innerArray ->
                betas[i] = betas[i].pow(2.0f)
            }
        } else {
            throw NotImplementedError("$betaSchedule is not implemented for ${this::class.java}")
        }

        alphas = betas.map { 1f - it }.toFloatArray()
        alphasCumprod = cumProd(alphas)

        sigmas = alphasCumprod.map { (sqrt((1.0f - it)/it)) }.toFloatArray()
        sigmas = concatenate(sigmas.reversedArray(),0.0f)

        initNoiseSigma = sigmas.max()

        timesteps = (linearSpace(0,numTrainTimesteps-1,numTrainTimesteps)).reversedArray()
        isScaleInputCalled = false
    }

    override fun scaleModelInput(sample: Array<Array<Array<FloatArray>>>, timestep: Int) : Array<Array<Array<FloatArray>>> {
        // val stepIndex = timesteps.indexOfFirst { it == timestep }
        val stepIndex =  nonzero(timesteps.map { if (it == timestep) 1.0f else 0.0f }.toFloatArray())
        val sigma = sigmas[stepIndex]
        var newSample = Array(sample.size) { Array(sample[0].size) { Array(sample[0][0].size) { FloatArray(sample[0][0][0].size) } } }
        sample.forEachIndexed { i, innerArray ->
            innerArray.forEachIndexed { j, innerInnerArray ->
                innerInnerArray.forEachIndexed { k, floatArray ->
                    floatArray.forEachIndexed { l, value ->
                        newSample[i][j][k][l] = sample[i][j][k][l] / (sqrt(sigma.pow(2.0f) + 1.0f))
                    }
                }
            }
        }
        isScaleInputCalled = true
        return newSample
    }

    override fun setTimesteps(numInferenceSteps: Int) {
        this.numInferenceSteps = numInferenceSteps
        timesteps = (linearSpace(0,numTrainTimesteps-1, numInferenceSteps)).reversedArray()
        sigmas = alphasCumprod.map { (sqrt((1f - it)/it)) }.toFloatArray()
        sigmas = interp( timesteps.map{ it.toFloat() }.toFloatArray(),arange(0f,sigmas.size.toFloat()),sigmas)
        sigmas = concatenate(sigmas,0.0f)

    }

    override fun step(modelOutput: Array<Array<Array<FloatArray>>>, timestep: Int, sample: Array<Array<Array<FloatArray>>>, eta: Float, useClippledModelOutput: Boolean, seed: Long?, varianceNoise: Array<Array<Array<FloatArray>>>?, returnDict: Boolean, imageCanvas: ImageView?): Any {
        val stepIndex = timesteps.indexOfFirst { it == timestep }
        val sigma = sigmas[stepIndex]
        var predOriginalSample = Array(modelOutput.size) { Array(modelOutput[0].size) { Array(modelOutput[0][0].size) { FloatArray(modelOutput[0][0][0].size) } } }
        val latentViewer = ORTImage()

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        when (predictionType) {
            "epsilon" -> sample.forEachIndexed { i, innerArray ->
                innerArray.forEachIndexed { j, innerInnerArray ->
                    innerInnerArray.forEachIndexed { k, floatArray ->
                        floatArray.forEachIndexed { l, value ->
                            predOriginalSample[i][j][k][l] = sample[i][j][k][l] - sigma *modelOutput[i][j][k][l]
                        }
                    }
                }
            }
            "v_prediction" -> sample.forEachIndexed { i, innerArray ->
                innerArray.forEachIndexed { j, innerInnerArray ->
                    innerInnerArray.forEachIndexed { k, floatArray ->
                        floatArray.forEachIndexed { l, value ->
                            predOriginalSample[i][j][k][l] = modelOutput[i][j][k][l] * (-sigma / sqrt(sigma.pow(2) + 1)) + (sample[i][j][k][l] / (sigma.pow(2) + 1))
                        }
                    }
                }
            }
            //predOriginalSample = modelOutput * (-sigma / sqrt(sigma.pow(2) + 1)) + (sample / (sigma.pow(2) + 1))
            else -> throw IllegalArgumentException("prediction_type given as $predictionType must be one of `epsilon`, or `v_prediction`")
        }
        //updateMainImageView(imageCanvas,latentViewer.imageFromLatents(predOriginalSample))
        val sigmaFrom = sigmas[stepIndex]
        val sigmaTo = sigmas[stepIndex + 1]
        val sigmaUp = sqrt((sigmaTo.pow(2) * (sigmaFrom.pow(2) - sigmaTo.pow(2)) / sigmaFrom.pow(2)))
        val sigmaDown = sqrt(sigmaTo.pow(2) - sigmaUp.pow(2))

        // 2. Convert to an ODE derivative
        //val derivative = (sample.zip(predOriginalSample){a,b->a-b} ).map { it/sigma }.toFloatArray()
        var derivative = Array(modelOutput.size) { Array(modelOutput[0].size) { Array(modelOutput[0][0].size) { FloatArray(modelOutput[0][0][0].size) } } }
        sample.forEachIndexed { i, innerArray ->
            innerArray.forEachIndexed { j, innerInnerArray ->
                innerInnerArray.forEachIndexed { k, floatArray ->
                    floatArray.forEachIndexed { l, value ->
                        derivative[i][j][k][l] =
                            (sample[i][j][k][l] - predOriginalSample[i][j][k][l]) / sigma
                    }
                }
            }
        }
        //updateMainImageView(imageCanvas,latentViewer.imageFromLatents(derivative))


        val dt = sigmaDown - sigma
        var prevSample = Array(modelOutput.size) { Array(modelOutput[0].size) { Array(modelOutput[0][0].size) { FloatArray(modelOutput[0][0][0].size) } } }
        sample.forEachIndexed { i, innerArray ->
            innerArray.forEachIndexed { j, innerInnerArray ->
                innerInnerArray.forEachIndexed { k, floatArray ->
                    floatArray.forEachIndexed { l, value ->
                        prevSample[i][j][k][l] = sample[i][j][k][l] + derivative[i][j][k][l] * dt
                    }
                }
            }
        }
        //updateMainImageView(imageCanvas,latentViewer.imageFromLatents(prevSample))
        val random = Random(seed!!+58917+timestep.toLong())
        val noise = Array(1) {
            Array(4) {
                Array(64) { index ->
                    FloatArray(64) { random.nextGaussian().toFloat()}
                }
            }
        }
        //updateMainImageView(imageCanvas,latentViewer.imageFromLatents(noise))
        //prevSample.mapIndexed{ index, sample -> prevSample[index] += noise[index] * sigmaUp }
        prevSample.forEachIndexed { i, innerArray ->
            innerArray.forEachIndexed { j, innerInnerArray ->
                innerInnerArray.forEachIndexed { k, floatArray ->
                    floatArray.forEachIndexed { l, value ->
                        prevSample[i][j][k][l] = prevSample[i][j][k][l] + noise[i][j][k][l] * sigmaUp
                    }
                }
            }
        }
        //updateMainImageView(imageCanvas,latentViewer.imageFromLatents(prevSample))

        return if (returnDict) EulerAncestralDiscreteSchedulerOutput(prevSample, predOriginalSample) else prevSample
    }



}