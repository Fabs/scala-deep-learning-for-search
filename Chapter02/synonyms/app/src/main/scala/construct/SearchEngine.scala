package construct

import concepts.LuceneConverter
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.core._
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper
import org.apache.lucene.index._
import org.apache.lucene.queryparser.classic._
import org.apache.lucene.search._
import org.apache.lucene.store._
import tools.tools.time

import java.nio.file.{Path, Paths}


class SearchEngine[T](val indexLocation: String, fieldAnalyzer: PerFieldAnalyzerWrapper, converter: LuceneConverter[T]) {
  val indexPath: Path = Paths.get(indexLocation)
  println(indexLocation)
  val directory: FSDirectory = FSDirectory.open(indexPath)
  val analyzers: PerFieldAnalyzerWrapper = fieldAnalyzer

  def index(docs: List[T], deleteAll: Boolean) = {
    val writerConfig = new IndexWriterConfig(analyzers)
    val writer = new IndexWriter(directory, writerConfig)

    if (deleteAll) {
      writer.deleteAll()
    }

    docs.foreach(doc => writer.addDocument(converter.toLuceneDoc(doc)))

    writer.commit()
    writer.close()
  }

  def search(queryText: String, field: String = "title", limit: Int = 100, analyzer: Analyzer = new WhitespaceAnalyzer(), explain: Boolean = false) = {
    val reader: IndexReader = DirectoryReader.open(directory)
    val queryParser = new QueryParser(field, analyzer)
    val query = queryParser.parse(queryText)
    val searcher: IndexSearcher = new IndexSearcher(reader)

    val docs: TopDocs = time("Search", {
      searcher.search(query, limit)
    })

    println(s"Total Hits: ${docs.totalHits}")
    println(query)
    docs.scoreDocs.foreach((scoreDoc) => {
      val doc = reader.document(scoreDoc.doc)
      println(s"> ${converter.fromLuceneDoc(doc)}")
      if(explain) {
        println(searcher.explain(query, scoreDoc.doc).toString)
      }
    })
    println("--------------------------------")
    reader.close()
  }
}
