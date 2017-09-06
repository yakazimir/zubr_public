"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Main setup file for installing/building sources and running
nose tests

"""
import os
import sys
import nose
import numpy
import re
import platform
from distutils.core import setup,Command
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.util import get_platform
from zubr.util import ConfigObj

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_DIR,"zubr")
TEST_DIR = os.path.join(SRC_DIR,"test")
AUX_DIR = os.path.join(PROJECT_DIR,"zubr/zubr_lisp")

# version
exec(open(os.path.join(PROJECT_DIR,'zubr/_version.py')).read())

## EXTRA MODULES, LIBRARIES


### EXTRA SETTINGS AND OPTIONAL DEPENDENCIES

OPTIONS = [
    ("--dynet","dynet","","str","Location of dynet library (optional)",""),
    ("--srilm","srilm","zubr/srilm","str","Location of srilm library (optional)",""),
    ("--arch","arch","i686-m64","str","Build architecture [default='i686-m4']",""),
    ("--eigen","eigen","/media/sf_projects/eigen","str","Location of eigen library (for dynet) [default='']",""),
    ("--boost","boost","/usr/include","str","Location of boost libraries (for dynet) [default='']",""),
]

## to add more dependencies, just add to DEP map below,
## where the key refers to the function that builds the cython or c extension
    
DEPS = {
    #"dynet" : setup_dynet,
    #"srilm" : setup_srilm,
}

DEP_CONFIG = ConfigObj(OPTIONS,{}).parse_known_args(sys.argv[1:])

## build_ext command

class build_zubr_ext(build_ext):

    """Build zubr extensions"""

    user_options = build_ext.user_options

    ## add the additional settings to build_ext
    for (name,_,_,ptype,descr,_) in OPTIONS:
        name = name if ptype == "bool" else name+"="
        name = re.sub(r'^\-+','',name)
        user_options.append((name,None,descr))

    def other_libraries(self):
        """Build other libaries specified in global class 

        :rype: list (of Extension modules) 
        """
        other_extensions = []

        for (dep_name,setup_fun) in DEPS.items():
            other_extensions += setup_fun(DEP_CONFIG)

        return other_extensions
            
    def initialize_options(self):
        build_ext.initialize_options(self)
        for opt in OPTIONS:
            setattr(self,opt[1],"")

    def build_extensions(self):
        ## build the extra libraries
        other = self.other_libraries()
        self.extensions += cythonize(other)

        ## sanity check the extensions again 
        self.check_extensions_list(self.extensions)

        ## build each extension
        for ext in self.extensions:
            ext.sources = self.cython_sources(ext.sources, ext)
            self.build_extension(ext)
        
# testing command

class TestCommand(Command):

    user_options = []

    def initialize_options(self):
        self._dir = PROJECT_DIR
        
    def finalize_options(self):
        pass
    
    def run_nose(self):
        test_dir = os.path.join(self._dir,TEST_DIR) 
        return nose.core.TestProgram(argv=[test_dir,"--exe"])
    
    def run(self):
        self.run_nose()

def check_for_build(file_path):
    if re.search(r'\#',file_path):
        return False
    elif '.pyx' in file_path and\
       not re.search(r'^C\_',file_path):
       return True
    return False

## standard cython code

CY_EXT = [os.path.join(SRC_DIR,x) for x in os.listdir(SRC_DIR) if check_for_build(x)]
CY_EXT += [os.path.join(AUX_DIR,x) for x in os.listdir(AUX_DIR) if check_for_build(x)]

## wrapped c code

## lz4 compression library 

LZ4_VERSION = (1, 3, 1, 2)
LZ4_VERSION_STR = ".".join([str(x) for x in LZ4_VERSION])

LZ4TOOLS = [
  Extension('zubr/lz4tools/lz4f', sources=[
            'zubr/lib/lz4.c',
            'zubr/lib/lz4hc.c',
            'zubr/lib/lz4frame.c',
            'zubr/lib/python-lz4f.c',
            'zubr/lib/xxhash.c'
        ], extra_compile_args=[
            "-std=c99",
            "-O3",
            "-Wall",
            "-W",
            "-Wundef",
            "-DVERSION=\"%s\"" % LZ4_VERSION_STR,
            "-DLZ4_VERSION=\"r124\"",
            ])]

### murmurhash3: https://github.com/hajimes/mmh3

HASH = [
    Extension('zubr/mmh3', sources=[
        'zubr/lib/mmh3module.cpp',
        'zubr/lib/MurmurHash3.cpp'
        ],
        language="c++",
    )]
    

if __name__ == "__main__":

        
    setup(name='zubr',
          version = __version__,
          cmdclass = {
              "build_ext":build_zubr_ext, ## switched from build_ext
              "test":TestCommand
            },
          author_email="kazimir.richardson@gmail.com",
          include_dirs=[numpy.get_include()],
          author="Kyle Richardson",
          url='https://www.github.com/yakazimir/zubr',
          description='Zubr: a python/cython semantic parsing library',
          platforms='any',
          scripts=['run_zubr','run_aligner'],
          packages=[
              'zubr',
              'zubr.util',
              'zubr.ex_scripts',
              'zubr.web',
              'zubr.zubr_lisp',
              'zubr.lz4tools',
              'zubr.doc_extractor',
            ],
          license="GPL >= 2",
          ## just take away LZ4TOOLS
          ext_modules=cythonize(CY_EXT+LZ4TOOLS+HASH), #+SRILM),
          classifiers= [
                "Programming Language :: Python",
                'Programming Language :: C',
                'Programming Language :: Cython',                
                "License :: GPL2",
                "Software :: Natural Language Processing",
                "Software :: Machine Learning",
                'Intended Audience :: Science/Research',
                'Intended Audience :: Developers',
                'Operating System :: MacOS',
                'Operating System :: Unix',
                ],
        )
