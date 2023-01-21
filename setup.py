from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


# `python setup.py build_ext --inplace`

setup(
    name="ocscore_lib",
    ext_modules=[
        Extension("ocscore",
            sources=["ocscore.pyx"],
            extra_compile_args=["-O3", "-mavx", "-mavx2"]
        )
    ],
    zip_safe=False,
    cmdclass = {'build_ext': build_ext}
)
