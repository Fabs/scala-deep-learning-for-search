import construct.{Analyzers, SearchEngine, W2VSynModel}
import data.SongData
import models.{Song, SongLuceneConverter}
import org.apache.lucene.analysis._
import org.apache.lucene.analysis.core._
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper
import synonyms.Config.{dataPath, indexPath}

import java.util

val docs = SongData.readSongDocuments(s"$dataPath/music.csv")
println(docs(1))

//We will be defining one Analyzer per Field
val perFieldAnalyzer = new util.HashMap[String, Analyzer]
val defaultAnalyzer = new EnglishAnalyzer()
perFieldAnalyzer.put("title", new WhitespaceAnalyzer())

val stopWords = EnglishAnalyzer.ENGLISH_STOP_WORDS_SET
perFieldAnalyzer.put("lyrics", new EnglishAnalyzer(stopWords))
val analyzers = new PerFieldAnalyzerWrapper(defaultAnalyzer, perFieldAnalyzer)

val engine = new SearchEngine[Song](s"$indexPath/music90", analyzers, new SongLuceneConverter )

engine.index(docs, deleteAll = true)

//engine.search("plane", field = "lyrics")
//engine.search("aeroplane", field = "lyrics")
//engine.search("airplane", field = "lyrics", explain = true)

//val w2vmodel = new W2VSynModel(s"$dataPath/music_training.txt")

//w2vmodel.near("plane", 10)
//w2vmodel.similar("rolling", accuracy = 0.90)

//val modelAnalyzer = Analyzers.synonymAnalyzerFromModel(s"$dataPath/music_training.txt", retrain = "false", minAccuracy = 0.90)
val analyzer = Analyzers.synonymAnalyzerFromModel(s"$dataPath/synonyms.txt")
engine.search("aeroplane", field = "lyrics", analyzer = analyzer)


//1. Create a basic search engine for songs
//2. Enhance the search with a list of synonyms on indexing
//2. Train a word2vec model
//3. Search with synonyms from wordnet at query time
