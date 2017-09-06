import os
import sys
from zubr.zubr_lisp.LispSetup import main as lsetup
from zubr.zubr_lisp.Reader import LispReader
from zubr.zubr_lisp.Shell import LispRepl
from zubr.zubr_lisp.LispTest import make_test
#from zubr.zubr_lisp.benchmark.benchmark import benchmark_core

curr_path = os.path.abspath(os.path.dirname(__file__))

def main(argv):
    """main function for launching zubr lisp"""
    config = lsetup(sys.argv)

    # run a test
    if config.test:
        make_test(config.test)
    ## launch shell
    # if config.benchmark:
    #     benchmark_core()
    elif config.repl is True:
        LispRepl()()
    else:
        exit('option not known, please see python -m zubr.zubr_lisp --help')

if __name__ == "__main__":
    main(sys.argv)

