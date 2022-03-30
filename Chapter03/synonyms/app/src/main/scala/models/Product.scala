package models

case class Product(line: Int, uid: Int, title: String, query: String) {
  def id(): String = {
    s"$uid"
  }

  override def toString(): String = {
    s"$title ($uid)"
  }
}
