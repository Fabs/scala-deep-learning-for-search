package construct

import data.SongData
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.core.{FlattenGraphFilter, FlattenGraphFilterFactory, WhitespaceTokenizerFactory}
import org.apache.lucene.analysis.custom.CustomAnalyzer
import org.apache.lucene.analysis.synonym.SynonymGraphFilterFactory
import org.deeplearning4j.models.word2vec.Word2Vec

object Analyzers {
  def synonymAnalyzerFromModel(trainDataFile: String, minAccuracy: Double = 0.90, retrain: String = "true"): Analyzer = {
    val options = new java.util.HashMap[String, String]
    options.put("trainDataFile", trainDataFile)
    options.put("minAccuracy", minAccuracy.toString)
    options.put("retrain", retrain)


    CustomAnalyzer.builder()
      .withTokenizer(WhitespaceTokenizerFactory.NAME)
      .addTokenFilter(classOf[W2VSynonymFilterFactory], options)
      .build()
  }

  def synonymAnalyzerFromFile(synonymFile: String): Analyzer = {
    val options = new java.util.HashMap[String, String]
    options.put("synonyms", synonymFile)
    options.put("ignoreCase", "true")

    CustomAnalyzer.builder()
      .withTokenizer(WhitespaceTokenizerFactory.NAME)
      .addTokenFilter(SynonymGraphFilterFactory.NAME, options)
      .addTokenFilter(FlattenGraphFilterFactory.NAME)
      .build()
  }
}
