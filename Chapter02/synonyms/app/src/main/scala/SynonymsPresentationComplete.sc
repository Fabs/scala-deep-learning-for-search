import construct.{Analyzers, SearchEngine, W2VSynModel}
import data.SongData
import models.{Song, SongLuceneConverter}
import org.apache.lucene.analysis._
import org.apache.lucene.analysis.core._
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper
import synonyms.Config.{dataPath, indexPath}

import java.util

//We will be defining one Analyzer per Field
val perFieldAnalyzer = new util.HashMap[String, Analyzer]
val defaultAnalyzer = new EnglishAnalyzer()

//Title Analyzer only breaks whitespace
perFieldAnalyzer.put("title", new WhitespaceAnalyzer())

val synList = s"$dataPath/synonyms.txt"
perFieldAnalyzer.put("title_syn1", Analyzers.synonymAnalyzerFromFile(synList))

//Lyrics Analyzer remove stop words
val stopWords = EnglishAnalyzer.ENGLISH_STOP_WORDS_SET
perFieldAnalyzer.put("lyrics", new EnglishAnalyzer(stopWords))

val analyzers = new PerFieldAnalyzerWrapper(defaultAnalyzer, perFieldAnalyzer)

val engine = new SearchEngine[Song](s"$indexPath/music4", analyzers, new SongLuceneConverter)

// Index musics from a csv, we delete all in every rerun
val songDataset = s"$dataPath/music.csv"
val docs = SongData.readSongDocuments(songDataset)
engine.index(docs, deleteAll = true)

engine.search("+plane")
engine.search("+aeroplane")
engine.search("+aeroplane", field = "title_syn1")

val trainingData = s"$dataPath/music_training.txt"
val synModel = new W2VSynModel(trainingData)
synModel.similar("guitar", 0.90)
synModel.similar("plane", 0.90)
synModel.similar("aeroplane", 0.90)
synModel.similar("rolling", 0.90)
synModel.similar("deep", 0.90)

val modelAnalyzer = Analyzers.synonymAnalyzerFromModel(trainingData, retrain = "false", minAccuracy = 0.90)
val queryAnalyzer = new PerFieldAnalyzerWrapper(modelAnalyzer, new util.HashMap[String, Analyzer])
//engine.search("+aeroplane", analyzer = queryAnalyzer, field = "lyrics")
//engine.search("+music +plane", field = "lyrics")
engine.search("+plane", analyzer = queryAnalyzer, field = "lyrics")
engine.search("+aeroplane", field = "lyrics")

//engine.search("+there +hot +heart", analyzer = queryAnalyzer, field = "lyrics", explain = true)
