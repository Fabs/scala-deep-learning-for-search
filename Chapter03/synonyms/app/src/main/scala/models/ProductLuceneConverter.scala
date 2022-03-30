package models

import concepts.LuceneConverter
import org.apache.lucene.document.{Document, Field, StoredField, TextField}

import scala.collection.mutable

class ProductLuceneConverter extends LuceneConverter[Product] {
  val docStore = new mutable.HashMap[String, Product]

  override def toLuceneDoc(src: Product): Document = {
    val luceneDoc = new Document
    luceneDoc.add(new StoredField("id", src.id))
    docStore.put(src.id, src)

    luceneDoc.add(new TextField("title", src.title, Field.Store.YES))

    luceneDoc
  }

  override def fromLuceneDoc(doc: Document): Option[Product] = {
    docStore.get(doc.get("id"))
  }
}
