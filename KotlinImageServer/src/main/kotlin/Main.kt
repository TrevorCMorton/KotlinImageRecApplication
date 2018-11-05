fun main(args : Array<String>) {
    val vgg16 = VGG16TinyImageNet()
    val model = vgg16.train()
}


