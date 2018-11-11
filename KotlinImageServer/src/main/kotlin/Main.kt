import org.datavec.image.loader.NativeImageLoader
import java.io.File

fun main(args : Array<String>) {
    val alexNet = AlexNetHotDogOrNot()
    val model = alexNet.train()
    val file = File("HotDog.jpg")
    val loader = NativeImageLoader(224, 224, 3)
    val image = loader.asMatrix(file);
    val scaler = alexNet.getScaler(alexNet.getIter(alexNet.trainDataPath))
    scaler.transform(image)
    val output = model.output(image)
    print(output)
}


