.onAttach <- function(libname, pkgname) {
    startUpMessage <- paste0("Package \"greybox\", v",packageVersion(pkgname)," loaded.");
    randomNumber <- trunc(runif(1,1,101));
    if(randomNumber<=4){
      # startUpMessage <- paste0(startUpMessage,"\n\033[38;2;00;70;20m");
      if(randomNumber==1){
        startUpMessage <- paste0(startUpMessage,"\nBy the way, have you already tried temporaldummy() function from greybox?");
      }
      else if(randomNumber==2){
        startUpMessage <- paste0(startUpMessage,"\nIf you want to know more about the greybox and forecasting, ",
                                 "you can visit my website: https://forecasting.svetunkov.ru/");
      }
      else if(randomNumber==3){
        startUpMessage <- paste0(startUpMessage,"\nAny thought or suggestions about the package? ",
                                 "Have you found a bug? File an issue on github: https://github.com/config-i1/greybox/issues");
      }
      else if(randomNumber==4){
        startUpMessage <- paste0(startUpMessage,"\nDid you know that you can use your own loss function in alm()? ",
                                 "This is regulated with 'loss' parameter. See documentation for examples.");
      }
    }
    startUpMessage <- paste0(startUpMessage,"\n");
    packageStartupMessage(startUpMessage);
}

.onUnload <- function (libpath) {
  library.dynam.unload("greybox", libpath);
}

# Function registers S3 method in the specific package
register_S3_method <- function(pkg, generic, class) {
  # Get the name of the proper function based on generic and class
  functionName <- get(paste0(generic, ".", class), envir=parent.frame());

  # Register the S3 methods if the package is already loaded
  if(pkg %in% loadedNamespaces()){
    registerS3method(generic, class, functionName, envir=asNamespace(pkg));
  }

  # Register hook on package load
  setHook(packageEvent(pkg, "onLoad"),
    function(...) {
      registerS3method(generic, class, functionName, envir=asNamespace(pkg))
    });
}

# Do this as a hook - if package is loaded, overwrite the function in it with the one from greybox
overwrite_S3_method <- function(pkg, generic){
  setHook(packageEvent(pkg, "onLoad"),
          function(...) {
            do.call("unlockBinding",list(generic,asNamespace(pkg)));
            assign(generic, get(generic, asNamespace("greybox")), envir=asNamespace(pkg));
            lockBinding(generic,asNamespace(pkg));
          },action="append");
}

.onLoad <- function(...) {
  # Do things if fabletools is present in the installed packages
  if(length(find.package("fabletools", quiet=TRUE, verbose=FALSE))!=0){
    overwrite_S3_method("fabletools","forecast");
    register_S3_method("fabletools","forecast","greybox");
    register_S3_method("fabletools","forecast","alm");
  }
  # Do things if forecast is present in the installed packages
  if(length(find.package("forecast", quiet=TRUE, verbose=FALSE))!=0){
    overwrite_S3_method("forecast","forecast");
    register_S3_method("forecast","forecast","greybox");
    register_S3_method("forecast","forecast","alm");
  }
  invisible();
}
