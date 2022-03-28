package models

import concepts.LuceneConverter
import org.apache.lucene.document.{Document, Field, StoredField, TextField}

import scala.collection.mutable

class SongLuceneConverter extends LuceneConverter[Song] {
  val docStore = new mutable.HashMap[String, Song]

  override def toLuceneDoc(src: Song): Document = {
    val luceneDoc = new Document
    luceneDoc.add(new StoredField("id", src.id))
    docStore.put(src.id, src)

    luceneDoc.add(new TextField("title", src.title, Field.Store.YES))
    luceneDoc.add(new TextField("lyrics", src.lyrics, Field.Store.YES))
    luceneDoc.add(new TextField("year", src.year.toString, Field.Store.YES))
    luceneDoc.add(new TextField("title_syn1", src.title, Field.Store.YES))

    luceneDoc
  }

  override def fromLuceneDoc(doc: Document): Option[Song] = {
    docStore.get(doc.get("id"))
  }
}
