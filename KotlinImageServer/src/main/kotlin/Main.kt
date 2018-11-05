import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

fun main(args : Array<String>) {
    val vgg16 = AlexNetTinyImageNet()
    var model : MultiLayerNetwork = vgg16.train()
}


