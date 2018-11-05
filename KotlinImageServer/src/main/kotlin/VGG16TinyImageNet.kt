import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.model.VGG16
import org.nd4j.linalg.learning.config.Nesterovs
import org.slf4j.LoggerFactory
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
            val inputShape = intArrayOf(3, 64, 64)
            val cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST
            val numClasses = 200
            val zooModel = VGG16.builder().inputShape(inputShape).numClasses(200).build()

            val model = zooModel.init()
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