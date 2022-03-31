import data.ProductData
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import synonyms.Config.dataPath
import tools.CharacterIterator

import java.io.PrintWriter
import java.nio.charset.StandardCharsets

val lstmLayerSize = 200
val sequenceSize = 77
val unrollSize = 10
val layers = new NeuralNetConfiguration.Builder()
  .list()
  .layer(0, new LSTM.Builder()
    .nIn(sequenceSize)
    .nOut(lstmLayerSize)
    .activation(Activation.TANH).build())
  .layer(1, new LSTM.Builder()
    .nIn(lstmLayerSize)
    .nOut(lstmLayerSize)
    .activation(Activation.TANH).build())
  .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
    .activation(Activation.SOFTMAX)
    .nIn(lstmLayerSize)
    .nOut(sequenceSize).build())
  .backpropType(BackpropType.TruncatedBPTT)
  .tBPTTForwardLength(unrollSize).tBPTTBackwardLength(unrollSize)
  .build();

val net = new MultiLayerNetwork(layers)
net.init()

val train = ProductData.readProductDocuments(s"$dataPath/product_train.csv")
val test = ProductData.readProductDocuments(s"$dataPath/product_test.csv")

new PrintWriter(s"$dataPath/queries.txt") {
  test.foreach(s => write(s.query + "\n"))
  train.foreach(s => write(s.query + "\n"))
  close
}

val miniBatchSize = 1
val train = new CharacterIterator(s"$dataPath/queries10.txt",
  StandardCharsets.UTF_8, miniBatchSize, sequenceSize)

net.setListeners(new ScoreIterationListener(1))
var i = 0
println(train.hasNext())
while (train.hasNext()) {
  net.fit(train)
  println(s"Batch $i - ${train.hasNext()}")
  i += 1
}

val initialization = "hero"
val input = Nd4j.zeros(sequenceSize, initialization.length())
val init = initialization.toCharArray()
(0 to init.length - 1).foreach(i => {
  val idx = train.convertCharacterToIndex(init(i))

  input.putScalar(Array(idx, i), 1.0f)
})

val output = net.rnnTimeStep(input)
println(output)
