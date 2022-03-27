package models

import org.apache.lucene.document.Field.Store
import org.apache.lucene.document.{Document, Field, StoredField, StringField, TextField}

import scala.collection.mutable

case class Song(rank: Int, title: String, artist: String, year: Int, lyrics: String, source: String) {
  def id(): String = {
    s"$title-$year"
  }

  override def toString(): String = {
    s"$title ($year) : ${lyrics.take(35)}"
  }
}
