from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "greybox._native_lowess",
        sources=["src/greybox/_native/lowess.cpp"],
        cxx_std=17,
    ),
    Pybind11Extension(
        "greybox._native_supsmu",
        sources=["src/greybox/_native/supsmu.cpp"],
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
