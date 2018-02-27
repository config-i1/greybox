.onAttach <- function(libname, pkgname) {
    packageStartupMessage(paste0("This is package \"geybox\", v",packageVersion(pkgname)));
}
