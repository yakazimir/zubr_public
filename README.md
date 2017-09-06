Zubr: A Semantic Parsing Toolkit 
==================

This is a slimmed down version of a cython/python toolkit for building
semantic parsing models (a more general release is in the works).

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

Inside of Zubr is a tool called `FunctionAssistant` that allows you to
build API query engines for source code collections. Below is an
example run involving the NLTK python2 project. 

    ## download a project, say tensorflow 
    bash-4.3$ get clone https://github.com/tensorflow/tensorflow

    ## train a model using build_query pipeline (this is a larger set..this took me around 50 minutes to train)
     bash-4.3$ ./run_zubr pipeline bin/build_query --proj /path/to/tensorflow/ --dir output/dir --aheuristic grow-diag --rlambda 0.0 --lrate1 0.001 --learner lsgd --eval_val --miters 5 --lang en --store_feat --class_info --online_addr /git/path/to/src
    
    ## launch the server
    bash-4.3$ python -m zubr.web.QueryServer --qmodels
    /output/dir/query --port 5000 


If you just want to extract a parallel dataset from an example API
(e.g.,  the one above), run the following with --stop:

     bash-4.3$ ./run_zubr pipeline bin/build_query --proj /path/to/tensorflow/ --dir output/dir --stop 2

Reproducing Experiments 
-----------------

In $ZUBR/experiments you will find scripts for downloading pipelines
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

Free software: GPL2 license. See LICENSE in this directory for more information. 
