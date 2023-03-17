package com.dynasoulstudio.jeremai

import android.graphics.Bitmap
import android.widget.ImageView
import androidx.core.math.MathUtils.clamp
import java.math.BigDecimal
import java.util.Random
import java.util.stream.IntStream.range
import kotlin.math.*

data class DDIMSchedulerOutput(val prevSample: Array<Array<Array<FloatArray>>>, val predOriginalSample: Array<Array<Array<FloatArray>>>)

internal class DDIMScheduler(
    val numTrainTimesteps: Int = 1000,
    val betaStart: Double = 0.0001,
    val betaEnd: Double = 0.02,
    val betaSchedule: String = "linear",
    val trainedBetas: DoubleArray? = null,
    val clipSample: Boolean = true,
    val setAlphaToOne: Boolean = true,
    val stepsOffset: Int = 0,
    val predictionType: String = "epsilon"
): SchedulerInterface {
    var betas: DoubleArray
    var alphas: DoubleArray
    var alphasCumprod: DoubleArray
    override var initNoiseSigma: Float = 0f
    var numInferenceSteps: Int? = null
    var isScaleInputCalled: Boolean
    override lateinit var timesteps: IntArray
    var finalAlphaCumprod: Double

    init {
        if (trainedBetas != null) {
            betas = trainedBetas
        } else if (betaSchedule == "linear") {
            betas = linearSpace(betaStart,betaEnd,numTrainTimesteps)
        } else if (betaSchedule == "scaled_linear") {
            betas = linearSpace(sqrt(betaStart),sqrt(betaEnd),numTrainTimesteps).map{it.pow(2.0)}.toDoubleArray()

        } else if (betaSchedule == "squaredcos_cap_v2") {
            // Glide cosine schedule
            betas = betasForAlphaBar(numTrainTimesteps)
        }else {
            throw NotImplementedError("$betaSchedule is not implemented for ${this::class.java}")
        }

        alphas = betas.map { 1.0 - it }.toDoubleArray()
        alphasCumprod = cumProd(alphas)

        finalAlphaCumprod = if(setAlphaToOne) 1.0 else alphasCumprod[0]

        initNoiseSigma = 1.0f

        timesteps = arange(0, numTrainTimesteps).reversedArray()

        isScaleInputCalled = false
    }

    fun betasForAlphaBar(numDiffusionTimesteps: Int,maxBeta: Double = 0.999): DoubleArray {
        fun alphaBar(timeStep: Double): Double{
            return (cos((timeStep + 0.008) / 1.008 * PI / 2).pow(2.0)).toDouble()
        }

        var betas = doubleArrayOf()
        for(i in  range(0,numDiffusionTimesteps)){
            val t1 = i.toDouble() /numDiffusionTimesteps
            val t2 = (i.toDouble()+1)/numDiffusionTimesteps
            betas+= min(1 - alphaBar(t2) / alphaBar(t1), maxBeta)
        }

        return betas
    }

    override fun scaleModelInput(sample: Array<Array<Array<FloatArray>>>, timestep: Int) : Array<Array<Array<FloatArray>>> {
        isScaleInputCalled = true
        return sample
    }

    fun _getVariance(
        timestep: Int,
        prevTimestep: Int
        ): Double{
        val alphaProdT = alphasCumprod[timestep]
        val alphaProdTPrev = if (prevTimestep >=0) alphasCumprod[prevTimestep] else this.finalAlphaCumprod
        val betaProdT = 1.0 - alphaProdT
        val betaProdTPrev = 1.0 - alphaProdTPrev

        val variance = (betaProdTPrev / betaProdT) * (1 - alphaProdT / alphaProdTPrev)

        return variance
    }

    override fun setTimesteps(numInferenceSteps: Int) {
        if (numInferenceSteps > numTrainTimesteps) {
            throw IllegalArgumentException("`num_inference_steps`: $numInferenceSteps cannot be larger than `self.config.train_timesteps`: ${numTrainTimesteps} as the unet model trained with this scheduler can only handle maximal ${numTrainTimesteps} timesteps.")
        }

        this.numInferenceSteps = numInferenceSteps
        val stepRatio = numTrainTimesteps.floorDiv(numInferenceSteps)
        timesteps = arange(0, numInferenceSteps).map{
            round(it * stepRatio.toDouble()).toInt()
        }.toIntArray().reversedArray()
        timesteps=timesteps.map{it+stepsOffset}.toIntArray()
    }

    override fun step(modelOutput: Array<Array<Array<FloatArray>>>, timestep: Int, sample: Array<Array<Array<FloatArray>>>, eta: Float, useClippledModelOutput: Boolean, seed: Long?, varianceNoise: Array<Array<Array<FloatArray>>>?, returnDict: Boolean, imageCanvas: ImageView?): Any {
        var predOriginalSample = Array(modelOutput.size) { Array(modelOutput[0].size) { Array(modelOutput[0][0].size) { FloatArray(modelOutput[0][0][0].size) } } }
        var modelOutput = modelOutput
        var varianceNoise = varianceNoise
        // See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        // Ideally, read DDIM paper in-detail understanding

        // 1. get previous step value (=t-1)
        val prevTimestep = timestep - numTrainTimesteps.floorDiv(numInferenceSteps!!)

        // 2. compute alphas, betas
        
        val alphaProdT = BigDecimal(alphasCumprod[timestep])
        val alphaProdTPrev = if (prevTimestep >=0) BigDecimal(alphasCumprod[prevTimestep]) else finalAlphaCumprod

        val betaProdT = (BigDecimal(1.0) - alphaProdT).toDouble()

        // 3. compute predicted original sample from predicted noise also called
        // "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        when (predictionType) {
            "epsilon" -> predOriginalSample.forEachIndexed { i, innerArray ->
                innerArray.forEachIndexed { j, innerInnerArray ->
                    innerInnerArray.forEachIndexed { k, doubleArray ->
                        doubleArray.forEachIndexed { l, value ->
                            predOriginalSample[i][j][k][l] = sqrt((sample[i][j][k][l] - sqrt(betaProdT) * modelOutput[i][j][k][l])/alphaProdT.toDouble()).toFloat()
                        }
                    }
                }
            }
            "sample" -> predOriginalSample = modelOutput
            "v_prediction" -> {
                predOriginalSample.forEachIndexed { i, innerArray ->
                    innerArray.forEachIndexed { j, innerInnerArray ->
                        innerInnerArray.forEachIndexed { k, doubleArray ->
                            doubleArray.forEachIndexed { l, value ->
                                predOriginalSample[i][j][k][l] =
                                    (sqrt(alphaProdT.toDouble()) * sample[i][j][k][l] - sqrt(
                                        betaProdT
                                    ) * modelOutput[i][j][k][l]).toFloat()
                            }
                        }
                    }
                }
                // predict V
                modelOutput.forEachIndexed { i, innerArray ->
                    innerArray.forEachIndexed { j, innerInnerArray ->
                        innerInnerArray.forEachIndexed { k, doubleArray ->
                            doubleArray.forEachIndexed { l, value ->
                                modelOutput[i][j][k][l] =
                                    (sqrt(alphaProdT.toDouble()) * modelOutput[i][j][k][l] + sqrt(
                                        betaProdT
                                    ) * sample[i][j][k][l]).toFloat()
                            }
                        }
                    }
                }
            }
            else -> throw IllegalArgumentException("prediction_type given as $predictionType must be one of `epsilon`, `sample` or `v_prediction` for the DDIMScheduler.")
        }

        // 4. Clip "predicted x_0"
        if(clipSample){
            predOriginalSample.forEachIndexed { i, innerArray ->
                innerArray.forEachIndexed { j, innerInnerArray ->
                    innerInnerArray.forEachIndexed { k, doubleArray ->
                        doubleArray.forEachIndexed { l, value ->
                            predOriginalSample[i][j][k][l] = clamp(predOriginalSample[i][j][k][l],-1.0f,1.0f)
                        }
                    }
                }
            }
        }

        // 5. compute variance: "sigma_t(η)" -> see formula (16)
        // σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        var variance = _getVariance(timestep,prevTimestep)
        var stdDevT = sqrt(eta * variance)

        if (useClippledModelOutput){
            // the model_output is always re-derived from the clipped x_0 in Glide
            modelOutput.forEachIndexed { i, innerArray ->
                innerArray.forEachIndexed { j, innerInnerArray ->
                    innerInnerArray.forEachIndexed { k, doubleArray ->
                        doubleArray.forEachIndexed { l, value ->
                            modelOutput[i][j][k][l] =
                                ((sample[i][j][k][l]-sqrt(alphaProdT.toDouble())*predOriginalSample[i][j][k][l]) / sqrt(betaProdT)).toFloat()
                        }
                    }
                }
            }
        }

        // 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        var predSampleDirection = Array(modelOutput.size) { Array(modelOutput[0].size) { Array(modelOutput[0][0].size) { FloatArray(modelOutput[0][0][0].size) } } }
        predSampleDirection.forEachIndexed { i, innerArray ->
            innerArray.forEachIndexed { j, innerInnerArray ->
                innerInnerArray.forEachIndexed { k, doubleArray ->
                    doubleArray.forEachIndexed { l, value ->
                        predSampleDirection[i][j][k][l] =
                            (sqrt((1.0 - alphaProdTPrev.toDouble() - sqrt(stdDevT))) * modelOutput[i][j][k][l]).toFloat()
                    }
                }
            }
        }

        // 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        var prevSample = Array(modelOutput.size) { Array(modelOutput[0].size) { Array(modelOutput[0][0].size) { FloatArray(modelOutput[0][0][0].size) } } }
        prevSample.forEachIndexed { i, innerArray ->
            innerArray.forEachIndexed { j, innerInnerArray ->
                innerInnerArray.forEachIndexed { k, doubleArray ->
                    doubleArray.forEachIndexed { l, value ->
                        prevSample[i][j][k][l] =
                            (sqrt(alphaProdTPrev.toDouble()) * predOriginalSample[i][j][k][l] + predSampleDirection[i][j][k][l]).toFloat()
                    }
                }
            }
        }

        if(eta > 0) {
            if (varianceNoise != null && seed != null) {
                throw IllegalArgumentException("Cannot pass both seed and variance_noise. Please make sure that either `seed` or `variance_noise` stays `null`.")
            }

            if (varianceNoise == null) {
                val random = Random(seed!!+58917+timestep)
                varianceNoise = Array(sample.size) {
                    Array(modelOutput[0].size) {
                        Array(modelOutput[0][0].size) {
                            FloatArray(modelOutput[0][0][0].size) { index ->
                                (random.nextGaussian().toFloat())
                            }
                        }
                    }
                }
            }
            var varianceArray = Array(modelOutput.size) {
                Array(modelOutput[0].size) {
                    Array(modelOutput[0][0].size) {
                        FloatArray(modelOutput[0][0][0].size)
                    }
                }
            }
            varianceArray.forEachIndexed { i, innerArray ->
                innerArray.forEachIndexed { j, innerInnerArray ->
                    innerInnerArray.forEachIndexed { k, doubleArray ->
                        doubleArray.forEachIndexed { l, value ->
                            varianceArray[i][j][k][l] =
                                (stdDevT * varianceNoise[i][j][k][l]).toFloat()
                        }
                    }
                }
            }
            prevSample.forEachIndexed { i, innerArray ->
                innerArray.forEachIndexed { j, innerInnerArray ->
                    innerInnerArray.forEachIndexed { k, doubleArray ->
                        doubleArray.forEachIndexed { l, value ->
                            prevSample[i][j][k][l] =
                                prevSample[i][j][k][l] + varianceArray[i][j][k][l]
                        }
                    }
                }
            }
        }


        return if (returnDict) DDIMSchedulerOutput(prevSample, predOriginalSample) else prevSample
    }


}