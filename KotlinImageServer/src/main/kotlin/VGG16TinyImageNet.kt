import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.learning.config.Nesterovs
import org.slf4j.LoggerFactory
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File

class VGG16TinyImageNet {

    private val log = LoggerFactory.getLogger(VGG16TinyImageNet::class.java)

    @Throws(Exception::class)
    fun train() : ComputationGraph {
        val modelFile = File("tinyvgg16.mod")
        if (modelFile.exists()){
            val model = ModelSerializer.restoreComputationGraph(modelFile)
            return model
        }
        else {
            //Get the DataSetIterators:
            val imNetTrain = TinyImageNetDataSetIterator(128)
            val imNetTest = TinyImageNetDataSetIterator(128)

            log.info("Build model....")
            val seed = 324L
            val updater = Nesterovs()
            val cacheMode = CacheMode.NONE
            val workspaceMode = WorkspaceMode.ENABLED
            val inputShape = longArrayOf(3, 64, 64)
            val cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST
            val numClasses = 200
            val conf =
                    NeuralNetConfiguration.Builder().seed(seed)
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                            .updater(updater)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.XAVIER)
                            .cacheMode(cacheMode)
                            .trainingWorkspaceMode(workspaceMode)
                            .inferenceWorkspaceMode(workspaceMode)
                            .graphBuilder()
                            .addInputs("in")
                            // block 1
                            .layer(0, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                    .cudnnAlgoMode(cudnnAlgoMode).build(), "in")
                            .layer(1, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "0")
                            .layer(2, SubsamplingLayer.Builder()
                                    .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                    .stride(2, 2).build(), "1")
                            // block 2
                            .layer(3, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "2")
                            .layer(4, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "3")
                            .layer(5, SubsamplingLayer.Builder()
                                    .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                    .stride(2, 2).build(), "4")
                            // block 3
                            .layer(6, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "5")
                            .layer(7, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "6")
                            .layer(8, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "7")
                            .layer(9, SubsamplingLayer.Builder()
                                    .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                    .stride(2, 2).build(), "8")
                            // block 4
                            .layer(10, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "9")
                            .layer(11, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "10")
                            .layer(12, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "11")
                            .layer(13, SubsamplingLayer.Builder()
                                    .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                    .stride(2, 2).build(), "12")
                            // block 5
                            .layer(14, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "13")
                            .layer(15, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "14")
                            .layer(16, ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                    .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "15")
                            .layer(17, SubsamplingLayer.Builder()
                                    .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                    .stride(2, 2).build(), "16")
                            .layer(18, DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                    .build(), "17")
                            .layer(19, DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                    .build(), "18")
                            .layer(20, OutputLayer.Builder(
                                    LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                                    .nOut(numClasses).activation(Activation.SOFTMAX) // radial basis function required
                                    .build(), "19")
                            .setOutputs("20")
                            .backprop(true).pretrain(false)
                            .setInputTypes(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                            .build();


            val model = ComputationGraph(conf)
            model.init()
            model.setListeners(ScoreIterationListener(5))  //print the score with every iteration

            log.info("Train model....")
            for (i in 0..15 - 1) {
                log.info("Epoch " + i)
                model.fit(imNetTrain)
            }


            log.info("Evaluate model....")
            val eval = Evaluation(numClasses)
            while (imNetTest.hasNext()) {
                val next = imNetTest.next()
                val output = model.output(next.features) //get the networks prediction
                //eval.eval(next.getLabels(), output) //check the prediction against the true class
            }

            log.info(eval.stats())
            log.info("****************Example finished********************")

            return model
        }
    }

}