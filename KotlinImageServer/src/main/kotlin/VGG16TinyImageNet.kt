import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.FlipImageTransform
import org.datavec.image.transform.ImageTransform
import org.datavec.image.transform.PipelineImageTransform
import org.datavec.image.transform.ResizeImageTransform
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.PretrainedType
import org.deeplearning4j.zoo.model.AlexNet
import org.deeplearning4j.zoo.model.LeNet
import org.deeplearning4j.zoo.model.VGG16
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Nesterovs
import org.slf4j.LoggerFactory
import java.io.File
import java.io.FilenameFilter
import java.util.*

class VGG16TinyImageNet {

    private val log = LoggerFactory.getLogger(VGG16TinyImageNet::class.java)

    @Throws(Exception::class)
    fun train() : ComputationGraph {
        val modelFile = File("hotdogvgg16.mod")
        if (modelFile.exists()){
            val model = ModelSerializer.restoreComputationGraph(modelFile)
            return model
        }
        else {
            //Get the DataSetIterators:

            log.info("Build model....")
            val batchSize = 128
            val seed = 324L
            val updater = Nesterovs()
            val cacheMode = CacheMode.NONE
            val workspaceMode = WorkspaceMode.ENABLED
            val cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST
            val numClasses = 2
            val zooModel = VGG16.builder().numClasses(2).build()

            val model = zooModel.init()
            model.init()
            model.setListeners(ScoreIterationListener(5))  //print the score with every iteration
            val inputShape = zooModel.metaData().getInputShape()[0]

            log.info("Load data....");
            val dataPath = "/home/trevor/Downloads/seefood/train"
            val fileSplit = FileSplit(File(dataPath), NativeImageLoader.ALLOWED_FORMATS, Random(324))
            val randomFilter = RandomPathFilter(Random(324), NativeImageLoader.ALLOWED_FORMATS, 0)

            log.info("Splitting data for production....");
            val split = fileSplit.sample(randomFilter, 0.8, 0.2)
            val trainData = split[0]
            val testData = split[1]
            log.info("Total training images is " + trainData.length())
            log.info("Total test images is " + testData.length())

            log.info("Calculating labels...")
            val file = File(dataPath)
            val directoryFilter = FilenameFilter( {current : File, name : String -> File(current, name).isDirectory()} )
            val directories = file.list(directoryFilter)

            log.info("Initializing RecordReader and pipelines....")
            var pipeline = LinkedList<Pair<ImageTransform, Double>>()
            pipeline.add(Pair(ResizeImageTransform(inputShape[1], inputShape[2]), 1.0))
            //pipeline.add(Pair(FlipImageTransform(1), 0.5))
            //val combinedTransform = PipelineImageTransform(324, pipeline as List<*>, false)
            val combinedTransform = PipelineImageTransform(ResizeImageTransform(inputShape[1], inputShape[2]))

            val labelMaker = ParentPathLabelGenerator()
            val trainRR = ImageRecordReader(inputShape[1].toLong(), inputShape[2].toLong(), inputShape[0].toLong(), labelMaker, combinedTransform)
            //trainRR.setLabels(directories.asList())
            trainRR.initialize(trainData)
            val testRR = ImageRecordReader(inputShape[1].toLong(), inputShape[2].toLong(), inputShape[0].toLong(), labelMaker, combinedTransform )
            //testRR.setLabels(directories.asList())
            testRR.initialize(testData)

            log.info("Total dataset labels: "+ directories.size)
            log.info("Total training labels: " + trainRR.getLabels().size)
            log.info("Total test labels: " + testRR.getLabels().size)

            log.info("Creating RecordReader iterator....")
            val trainIter = RecordReaderDataSetIterator(trainRR, batchSize, 1, 2)
            val testIter = RecordReaderDataSetIterator(testRR, batchSize, 1, 2)

            testIter.next()

            log.info("Fitting normalizer...")
            val scaler = ImagePreProcessingScaler(0.0, 1.0)
            scaler.fit(trainIter)
            trainIter.setPreProcessor(scaler)
            testIter.setPreProcessor(scaler)

            log.info("Train model....")
            for (i in 0..15 - 1) {
                log.info("Epoch " + i)
                model.fit(trainIter)
            }


            log.info("Evaluate model....")
            val eval = Evaluation(numClasses)
            model.evaluate(testIter);
            //while (testIter.hasNext()) {
            //    val next = testIter.next()
            //    val output = model.output(next.features) //get the networks prediction
            //    eval.eval(next.labels, output) //check the prediction against the true class
            //}

            log.info(eval.stats())
            log.info("****************Example finished********************")

            ModelSerializer.writeModel(model, modelFile, false)

            return model
        }
    }

}