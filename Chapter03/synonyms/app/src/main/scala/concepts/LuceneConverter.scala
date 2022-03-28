package concepts

import org.apache.lucene.document.Document

trait LuceneConverter[T] {
  def toLuceneDoc(src: T): Document

  def fromLuceneDoc(doc: Document): Option[T]
}
