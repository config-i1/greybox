.onAttach <- function(libname, pkgname) {
    packageStartupMessage(paste0("Package \"greybox\", v",packageVersion(pkgname)," loaded."));
}

.onUnload <- function (libpath) {
  library.dynam.unload("greybox", libpath)
}
