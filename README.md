Zubr: A Semantic Parsing Toolkit (Version 2.0)
==================

This is a cython/python toolkit for building semantic parsing
models (with a focus on certain types of text-to-code translation models). See references for related work. 

Please use at your own risk, this is academic code and not well tested
(and in some cases quite hacky).

Quick Start 
-----------------

To build zubr locally, do:

    make build-ext 

Then run the parser using the ``run_zubr'' script, where mode is the
type of task you want to run. 

    ./run_zubr mode [options]

Typing the following will show you the different modes you can run:

    ./run_zubr --help 

See inside of the ``run_zubr'' script to see how to call directly from
python.

This library also includes a small implementation of lisp called ``zubr_lisp``,
which can be evoked by running the following script:

    ./zubr_lisp [options] 

Details of the lisp implementation are included in zubr/zubr_lisp. 

Installation (not recommended)
-----------------

To do a system wide installation (after building the sources using the
command above), do:

    python setup.py install 

Alternatively, to install locally using virtualenv
(https://virtualenv.pypa.io/en/latest/)  do : 

    make build-local

(assumes that virtualenv is already installed. This will install all
required modules/packages listed in requirements.txt)

For running simple experiments, or reproducing previous
experiments, we recommend building locally, as described in the
QuickStart.

Graph Decoder and Neural Models
--------------------------------

As of Version 2.0, we are including the code used for training various
types of graph semantic parsing models, as reported in Richardson et
al. NAACL 2018 (see references).

Installing these modules requires first installing Dynet (for
building neural network models)
(https://github.com/clab/dynet, version 2.0 (very important!), a zipped version is
included here, following installation instructions there) and FOMA (https://fomafst.github.io/, the binaries for Mac and Linux are
included in this distribution in bin/foma).

Once you have installed Dynet (and its dependencies, e.g., Eigen and
Boost) and Foma, be sure to set the environment variables ``$DYNET,
$EIGEN, $BOOST``. With these variables set, the ``make`` command above
should do the rest.

We also have code for running ``executable'' semantic parsers (e.g.,
in the GeoQuery domain), which will require having prolog installed (https://sicstus.sics.se/). 

See details in ``experiments/`` to see how to run these models. 

Modes and Pipelines
-----------------

Running Zubr in a particular mode will allow you to use a particular
Zubr utility independent of all others. For example, the following
runs the aligner mode, which will train an end-to-end alignment model
on a portion of the hansards corpus. 

    ./run_zubr alignment --atraining examples/alignment/hansards --modeltype ibm1 --aligntraining 

To see the full list of modes, see $ZUBR/zubr/__main__ , or just type
./run_zubr to get more information. 

Zubr can be run in a ``pipeline'' mode, which allows you to combine
multiple zubr utilities and interface them with other external
functions or code. The aim is to make it easy to quickly setup and run
different experiments using zubr tools. 

Each pipeline must include a pipeline script (in python). For example:

    ./run_zubr pipeline bin/ex_pipeline.py --atraining examples/aligner/hansards --dir examples/_sandbox/ex_pipeline

will run a pipeline calling the script ``ex_pipeline``, which might
look like (in normal python):

```python

# import any other python module in the normal fashion

## script parameters
## format: cli switch, switch name, default val, type, description, script name
params = [
    ("--param1","param1",False,"bool","description of param1","MyScript"),
    ("--param2","param2",False,"bool","description of param2","MyScript")
]

## name and description of script ( script name -> description)
description = {"MyScript":"settings for my script"}

## tasks to perform in pipeline (in order)
## anything with zubr.X will run module X in zubr toolkit
tasks = [
    "setup_pipeline",
    "zubr.Alignment",
    "close_pipeline",
]

## all script functions should have config as first argument (regardless of
## whether they use a config or not)

def setup_pipeline(config):
    ## function at beginning of script/pipeline, e.g., building data 
    print "setting up pipeline, building data"

def close_pipeline(config):
    ## function for end of script/pipeline, e.g., evaluating results
    print "closing the pipeline, evaluate results,..."
    
```

This pipeline script will run a zubr aligner (using settings set in
configuration) in-between two custom functions (e.g., the first
function might be involved in building the data for the aligner, and
the second for evaluating the output). Other python modules or
functions can be used in the script as needed. 

Running the following will show all of the configuration parameters for this
script for all of the zubr utilities being used:

    ./run_zubr pipeline bin/ex_pipeline --help 

Once you plug in the desired configuration parameters, the pipeline
will take care of the rest.

Building an API query engine
-----------------

Inside of Zubr is a tool called `FunctionAssistant`  that allows you
to build API query engines for source code (currently Python and Java) collections. Below is an
example run involving the Pyglet project. 

 ```
     ## 
     mkdir examples/codelibs
     cd examples/codelibs

     ## download a project (e.g., Pyglet, really small project, won't have too much to query, but demonstrates how pipeline works) 
     git clone git@github.com:adamlwgriffiths/Pyglet.git
     cd ../../

     ## extract data, train model, build query object (add preferred settings accordingly)
    ./run_zubr pipeline bin/build_server --proj examples/codelibs/Pyglet --dir examples/pyglet_exp --aheuristic grow-diag --lrate 0.001 --eval_val --miters1 5 --store_feat --class_info --online_addr  https://github.com/adamlwgriffiths/Pyglet/tree/master/pyglet --src_loc pyglet

    ## launch the server 
    ./run_zubr queryserver --qmodels examples/pyglet_exp/query --port 5000

 ```

If you just want to extract a parallel dataset from an example API
(e.g.,  the one above), run the following with --end:

     bash-4.3$ ./run_zubr pipeline bin/build_query --proj /path/to/tensorflow/ --src_loc tensorflow --dir output/dir --end 2


Please note that when building datasets and experiment splits, the way
the split is done appears to be specific to each os. So running this
on different computers might get different results. It's therefore
much better to build a single dataset first, then run experiments on a
set of static files. 

Please also note that the extractor might miss a lot of documentation,
and you might need to massage the src directory structure to point it
to all the data.

If you use this tool, please cite the following:

```
@inproceedings{richardson-kuhn:2017:Demo,
  author    = {Richardson, Kyle  and  Kuhn, Jonas},
  title     = {Function {A}ssistant: {A} {T}ool for {NL} {Q}uerying of {API}s},
  booktitle = {Proceedings of the EMNLP},
  year      = {2017},
  }
```

Reproducing Experiments 
-----------------

In experiments/ you will find scripts for downloading pipelines
for various reported results.

Development
-----------------

To run the all tests, do the following (assumes nose, this will call
python setup.py test):

    make test 

Author
----------------

Kyle Richardson (University of Stuttgart)

*kyle@ims.uni-stuttgart.de*

License 
----------------

Free software: GPL2 license. See LICENSE in this directory for more
information. For the various dependencies (e.g., Dynet, Foma, ...)
please observe the licenses provided there. 

References
----------------

This work is based on the following work (please cite the EMNLP paper
if you use the tool): 

```
@inproceedings{richardson-naacl,
  author    = {Richardson, Kyle, Berant,Jonathan  and  Kuhn, Jonas},
  title     = {Polyglot {S}emantic {P}arsing in {API}s},
  booktitle = {Proceedings of the NAACL},
  year      = {2018},
  url        ={https://arxiv.org/abs/1803.06966},
  }

@inproceedings{richardson-kuhn:2017:Long,
  author    = {Richardson, Kyle  and  Kuhn, Jonas},
  title     = {Learning {S}emantic {C}orrespondences in {T}echnical {D}ocumentation},
  booktitle = {Proceedings of the ACL},
  year      = {2017},
  url={http://aclweb.org/anthology/P/P17/P17-1148.pdf},
  }

@inproceedings{richardson-kuhn:2017:Demo,
  author    = {Richardson, Kyle  and  Kuhn, Jonas},
  title     = {Function {A}ssistant: {A} {T}ool for {NL} {Q}uerying of {API}s},
  booktitle = {Proceedings of the EMNLP},
  year      = {2017},
  url={https://arxiv.org/abs/1706.00468},
  }
```
