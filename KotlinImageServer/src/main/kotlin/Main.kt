import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.net.ServerSocket

fun main(args : Array<String>) {
    val alexNet = AlexNetHotDogOrNot()
    val model = alexNet.train()
    while(true) {
        val ss = ServerSocket(10000)
        println("Waiting for client")
        val socket = ss.accept()
        println("accepted client")
        val rawOutput = socket.getOutputStream()
        val outputStream = ObjectOutputStream(rawOutput)
        val inputStream = ObjectInputStream(socket.getInputStream())

        val imageFileBytes = inputStream.readObject() as ByteArray
        println("writing image to file")
        val imageFile = File("temp.jpg")
        imageFile.writeBytes(imageFileBytes)

        val file = File("temp.jpg")
        val loader = NativeImageLoader(224, 224, 3)
        val image = loader.asMatrix(file)
        println("Interpreted image")
        val scaler = alexNet.getScaler(alexNet.getIter(alexNet.trainDataPath))
        scaler.transform(image)
        val output = model.output(image)
        print(output.getFloat(0))
        print(output.getFloat(1))
        if (output.getFloat(0) > output.getFloat(1)) {
            println("Hot dog!")
            outputStream.writeObject("Hot Dog")
        } else {
            println("Not Hot Dog...")
            outputStream.writeObject("Not Hot Dog")
        }
        ss.close()
        println("Disconnected from client")
    }
}


