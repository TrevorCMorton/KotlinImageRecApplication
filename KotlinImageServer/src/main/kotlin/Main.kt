import org.deeplearning4j.nn.graph.ComputationGraph

fun main(args : Array<String>) {
    val vgg16 = VGG16TinyImageNet()
    var model : ComputationGraph = vgg16.train()
}


