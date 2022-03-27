package tools

package object tools {
  def time[R](message: String, block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println(s"$message: ${(t1 - t0) / 1000000.0}ms")
    result
  }
}
