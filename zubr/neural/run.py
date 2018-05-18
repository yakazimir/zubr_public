import logging
import sys
from zubr.neural import start_dynet
from zubr.neural import _dynet as dy
from zubr.util.config import ConfigObj,ConfigAttrs

## module parameters
from zubr.neural.Seq2Seq import params as sparams
from zubr.neural.ShortestPathDecoder import params as dparams
from zubr.neural.util import params as uparams

## program modes
from zubr.neural.Seq2Seq import run_seq2seq
from zubr.neural.ShortestPathDecoder import run_decoder


def params():
    """The parameters for running this neural stuff 

    
    """
    options = [
        ("--mem","mem",512,int,
         "Dynet memory allocation [default=90]","GeneralNeural"),
        ("--seed","seed",2798003128,int,
         "Dynet random seed [default=2798003128]","GeneralNeural"),
        ("--neural_toolkit","neural_toolkit","dynet","str",
         "The toolkit to use [default='dynet']","GeneralNeural"),
        ## trainer type 
        ("--trainer","trainer","sgd","str",
         "The type of model trainer to use [default='sgd]","GeneralNeural"),
        ## various hyper parameters 
        ("--epochs","epochs",10,int,
         "The number of training epochs [default=10]","GeneralNeural"),
        ("--lrate","lrate",0.1,float,
         "The trainer learning rate [default=0.1]","GeneralNeural"),
        ("--weight_decay","weight_decay",0.0,float,
         "The main weight decay parameter [default=0.0]","GeneralNeural"),
        ("--epsilon","epsilon",1e-20,float,
         "The epsilon parameter [default=1e-20]","GeneralNeural"),
        ("--momentum","momentum",0.9,float,
         "The momentum parameter [default=0.9]","GeneralNeural"),
        ## the run mode 
        ("--mode","mode","decoder","str",
         "The type of mode to run [default='decoder']","GeneralNeural"),
        ("--alpha","alpha",0.001,float,
         "The initial learning rate for adam [default=0.001]","GeneralNeural"),
        ("--beta_1","beta_1",0.9,float,
         "The moving average parameter for mean [default=0.9]","GeneralNeural"),
        ("--beta_2","beta_2",0.999,float,
         "The Moving average for variable [default=0.999]","GeneralNeural"),
        ("--eps","eps",1e-8,float,
         "epsilon for adam [default=1e-8]","GeneralNeural"),
        ("--nmodel","nmodel",'',"str",
         "The location of the model (if exists) [default='']","GeneralNeural"),
        ("--dropout","dropout",0.0,"float",
         "The dropout rate (shut off by default) [default=0.0]","GeneralNeural"),
    ]

    group = {"GeneralNeural" : "General neural network settings"}

    ## seq2seq model parameters
    sgroup,soptions = sparams()
    options += soptions
    group.update(sgroup)

    ## utility parameters
    ugroup,uoptions = uparams()
    options += uoptions
    group.update(ugroup)

    ## decoder parameters
    dgroup,doptions = dparams()
    options += doptions
    group.update(dgroup)
    
    return (group,options) 

def argparser():
    """Build an argument parser for using this module


    """
    from zubr import _heading
    from _version import __version__ as v
    
    usage = """python -m zubr neural [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage)
    return argparser


def main(argv):
    """Main execution point for running neural models 

    :param argv: arguments to neural code
    """
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])
        #if not config.jlog: logging.basicConfig(level=logging.DEBUG)


    ## use dynet
    if config.neural_toolkit == "dynet":
        
        ## initialize dynet memory and random seed 
        start_dynet(config,dy.DynetParams())

        ##
        if config.mode == "decoder":
            run_decoder(config)
        else: 
            run_seq2seq(config)
    
if __name__ == "__main__":
    main(sys.arv[1:])
