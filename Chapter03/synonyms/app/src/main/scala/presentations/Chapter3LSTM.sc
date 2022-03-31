import construct.{ExpandQuery, SearchEngine}
import data.ProductData
import models.{Product, ProductLuceneConverter}
import org.apache.lucene.analysis._
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper
import synonyms.Config.{dataPath, indexPath}

import java.io.PrintWriter
import java.util

val perFieldAnalyzer = new util.HashMap[String, Analyzer]
val defaultAnalyzer = new EnglishAnalyzer()
val analyzers = new PerFieldAnalyzerWrapper(defaultAnalyzer, perFieldAnalyzer)

val engine = new SearchEngine[Product](s"$indexPath/index",
  analyzers, new ProductLuceneConverter)

val train = ProductData.readProductDocuments(s"$dataPath/product_train.csv")
engine.index(train, deleteAll = true)
val test = ProductData.readProductDocuments(s"$dataPath/product_test.csv")
engine.index(test, deleteAll = true)

engine.search("shower stuff", analyzer = defaultAnalyzer, limit = 5)
engine.search("shower", analyzer = defaultAnalyzer,
  queryParser = new ExpandQuery("title", defaultAnalyzer), limit = 5)
