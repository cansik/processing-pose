package ch.fhnw.exakt.util

object OSValidator {
    private val OS = System.getProperty("os.name").toLowerCase()

    val isWindows: Boolean
        get() = OS.indexOf("win") >= 0

    val isMac: Boolean
        get() = OS.indexOf("mac") >= 0

    val isUnix: Boolean
        get() = OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0

    val isSolaris: Boolean
        get() = OS.indexOf("sunos") >= 0
}