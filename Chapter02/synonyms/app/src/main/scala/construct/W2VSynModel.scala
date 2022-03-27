package construct

import data.SongData
import org.deeplearning4j.models.embeddings.learning.impl.elements.{CBOW, SkipGram}
import org.deeplearning4j.models.word2vec.{VocabWord, Word2Vec}
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator

import scala.collection.JavaConverters._
import java.nio.file.Paths

class W2VSynModel(trainingDataFile: String, retrain: Boolean = true) {
  if(retrain || W2vSynModelSingleton.model == null) {
    val trainingData = Paths.get(trainingDataFile)
    val dataIter = new BasicLineIterator(trainingData.toFile)
    val modelInTraining = new Word2Vec.Builder()
      .layerSize(300)
      .windowSize(5)
      .iterate(dataIter)
      .elementsLearningAlgorithm(new SkipGram[VocabWord]())
      .build()

    modelInTraining.fit()

    W2vSynModelSingleton.model = modelInTraining
  }

  val model = W2vSynModelSingleton.model

  def near(word: String, limit: Int = 2): List[String] = {
    model.wordsNearest(word, limit).asScala.toList
  }

  def similar(word: String, accuracy: Double) = {
    model.similarWordsInVocabTo(word, accuracy).asScala.toList
  }
}

object W2vSynModelSingleton {
  var model:Word2Vec = null
}
