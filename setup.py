#from distutils.core import setup, Extension
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy
ext_modules = cythonize(
        [Extension('em_probit._sampling',
                   ['em_probit/_sampling.pyx'],
                   include_dirs = [numpy.get_include()] )])


setup(
    packages=['em_probit'],
    #ext_modules = cythonize("_sampling.pyx",include_path = [numpy.get_include()])
    ext_modules = ext_modules,
    cmdclass={"build_ext": build_ext}
)