import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.*
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.model.AlexNet
import org.nd4j.jita.conf.CudaEnvironment
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Nesterovs
import org.slf4j.LoggerFactory
import java.io.File
import java.io.FilenameFilter
import java.util.*

class AlexNetHotDogOrNot {

    private val log = LoggerFactory.getLogger(AlexNetHotDogOrNot::class.java)

    val trainDataPath = "C:\\Users\\trevo\\Downloads\\seefood\\train"
    val testDataPath = "C:\\Users\\trevo\\Downloads\\seefood\\test"

    val batchSize = 64
    val seed = 324L
    val numClasses = 2
    val inputShape = listOf(3, 224, 224)

    @Throws(Exception::class)
    fun train() : MultiLayerNetwork {
        val modelFile = File("alexnethotdog.mod")
        if (modelFile.exists()){
            val model = ModelSerializer.restoreMultiLayerNetwork(modelFile)
            return model
        }
        else {
            //Get the DataSetIterators:

            CudaEnvironment.getInstance().configuration.setMaximumDeviceCache(6L * 1024L * 1024L * 1024L)

            log.info("Build model....")

            val zooModel = AlexNet.builder().numClasses(this.numClasses).updater(Nesterovs(5e-3, 0.9)).build()

            val model = zooModel.init()
            model.setListeners(ScoreIterationListener(5))  //print the score with every iteration

            log.info("Load data....")
            val trainIter = this.getIter(this.trainDataPath)

            val testIter = this.getIter(this.testDataPath)

            log.info("Fitting normalizer...")

            val scaler = this.getScaler(trainIter)

            trainIter.setPreProcessor(scaler)
            testIter.setPreProcessor(scaler)

            log.info("Train model....")
            for (i in 0..300) {
                log.info("Epoch " + i)
                model.fit(trainIter)
                log.info(model.evaluate(testIter).stats())
            }

            log.info("****************Example finished********************")

            ModelSerializer.writeModel(model, modelFile, true)

            return model
        }
    }

    fun getIter(directoryName : String) : RecordReaderDataSetIterator {
        val fileSplit = FileSplit(File(directoryName), NativeImageLoader.ALLOWED_FORMATS, Random(seed))
        val randomFilter = RandomPathFilter(Random(seed), NativeImageLoader.ALLOWED_FORMATS, 0)

        log.info("Splitting data for production....");
        val split = fileSplit.sample(randomFilter, 1.0, 0.0)
        val data = split[0]
        log.info("Total images is " + data.length())

        log.info("Calculating labels...")
        val file = File(directoryName)
        val directoryFilter = FilenameFilter( {current : File, name : String -> File(current, name).isDirectory()} )
        val directories = file.list(directoryFilter)

        log.info("Initializing RecordReader and pipelines....")
        val resizeTransform : ImageTransform = ResizeImageTransform(this.inputShape[1], this.inputShape[2])
        val flipTransform1 : ImageTransform = FlipImageTransform(Random(seed))
        val flipTransform2 : ImageTransform  = FlipImageTransform(Random(123))
        val warpTransform : ImageTransform = WarpImageTransform(Random(324), 42f)
        val shuffle = false
        val pipeline : List<org.nd4j.linalg.primitives.Pair<ImageTransform, Double>> = listOf(org.nd4j.linalg.primitives.Pair(resizeTransform,1.0), org.nd4j.linalg.primitives.Pair(flipTransform1,0.5), org.nd4j.linalg.primitives.Pair(flipTransform2,0.5), org.nd4j.linalg.primitives.Pair(warpTransform,0.5))
        val combinedTransform = PipelineImageTransform(324L, pipeline, shuffle)

        val labelMaker = ParentPathLabelGenerator()
        val rr = ImageRecordReader(this.inputShape[1].toLong(), this.inputShape[2].toLong(), this.inputShape[0].toLong(), labelMaker, combinedTransform)
        rr.initialize(data)

        log.info("Total dataset labels: "+ directories.size)
        log.info("Total labels: " + rr.getLabels().size)

        log.info("Creating RecordReader iterator....")
        val iter = RecordReaderDataSetIterator(rr, this.batchSize, 1, this.numClasses)

        return iter
    }

    fun getScaler(iter : RecordReaderDataSetIterator) : ImagePreProcessingScaler {
        val scaler = ImagePreProcessingScaler(0.0, 1.0)
        scaler.fit(iter)

        return scaler
    }

}