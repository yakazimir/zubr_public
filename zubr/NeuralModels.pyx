# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementation of various kinds of neural language models using
different frameworks (currently, mostly via Dynet) 

"""
import os
import sys
import logging
import traceback
import time
import numpy as np
cimport numpy as np
from zubr.util.config import ConfigAttrs
from zubr.ZubrClass cimport ZubrSerializable
from zubr.util.alignment_util import load_aligner_data
from zubr.Alg cimport binary_insert_sort
from cython cimport boundscheck,wraparound,cdivision


### models

cdef class NeuralModel(ZubrSerializable):
    """Base class for neural models, defines methods for 
    training and testing using the models.     
    """

    @classmethod
    def from_config(cls,config):
        """Build neural model from configuration 

        :param config: the main configuration
        :returns: a Neural model instance 
        """
        raise NotImplementedError()

    cpdef void train(self,config):
        """Train the neural network model using configuration

        :param config: the configuration with pointers to data, other info,..
        :returns: None 
        """
        raise NotImplementedError()

## dynet neural models 

cdef class DynetModels(NeuralModel):
    """This is baseclass for various dynet neural network architectures.

    It assumes that dynet is compiled and installed, including (importantly) 
    the provided python/cython wrapper code. For more details, see the below: 

    https://github.com/clab/dynet

    Note: these classes currently interact with dynet using Python, and not 
    Cython (which would make more sense, and would probably be more efficient). 
    This is because I haven't yet started looking into the dynet source (cython
    and otherwise) yet. Depending which types of architectures work well, one 
    possibly direction is writing my own Cython extensions specific to these 
    particular architectures, using of course the existing Cython as a guide.
    """

    @classmethod
    def _load_dynet(cls,config):
        """Loads the dynet module, raises exception if not found, and creates 
        the desired models and trainers 

        :param config: the global configuration object 
        :returns: tuple of dynet components
        :raises: ImportError,ValueError 
        """
        raise NotImplementedError()

cdef class DynetFeedForward(DynetModels):
    """Base class for defining dynet FeedForward network architectures"""

    @classmethod
    def _load_dynet(cls,config):
        """Loads the dynet module, raises exception if not found 

        :param config: the global configuration object 
        :returns: tuple of dynet components
        :raises: ImportError,ValueError 
        """
        ## load the dynet module, a model, and a trainer (respectively0
        dy = __find_dynet(config)
        model = dy.Model()

        return (dy,model)


cdef class FeedForwardLM(DynetFeedForward):
    """A Feed forward language model"""
    pass
    
cdef class FeedForwardTranslationLM(FeedForwardLM):

    """Concrete Implementation for a shallow feed-forward translation language model

    How this works: These models are similar to standard FeedForward language models, with
    the difference that each word w_{j} in some (target) language, let's call it f, is condition by 
    the existence of a source language, call it e, in addition to the ngram context, w_{j-1},w_{j-2},...

    As such, the model is used for translation, or learning the distribution: p(f | e), which is computed 
    by computing the product of each word translation in f. Note that this model is not a decoder, and 
    therefore assumes the existence of a full e and f. In our case, the idea is to use this model in a 
    graph-based decoder that knows about the structure of the f language in order to make local transition
    decisions. 

    Required parameters: 
    
        representation of n-gram contexts: Standardly these are represented as word embeddings, the 
        dimensions of which can be set using with --embed_dim 

        representation of e: The english sentence has a variable length, but in this model we assume 
        a maximum input size --epos, and represent each position in this size using a word embedding, 
        the dimension of which is specified using --e_embed_dim. Padding is used in cases where the 
        input size is less than 30 

        length parameters: the length of the english sentence, which is representation using an embedding 
        using a size set by --elen_embed_dim, and the current length of each f item expressed as an embedding 
        set using --flen_embed_dim 

    Optional parameters: 

        word translation pairs : have explicit (w_{e},w_{f}) pair features. As done elsewhere in Zubr, we 
        can create a unique identifier for each pair by doing the following: pair_id = |f vocab |*f_id + e_id. 
        Now, for large vocabularies, the space of pairs gets really large. To avoid this, and also use emebeddings, 
        we use a hash trick, as done in Botha et al cited above, to reduce these identifiers to a small set, then 
        associate with each index a small embedding (10-50). 

        bigram-bigram prairs : 

    """

    def __init__(self,dynet,
                     model,
                     trainer,
                     context_embed,
                     e_embed,
                     len_embed,
                     flen_embed,
                     hlist,hbias,
                     olayer,obias,
                     elex,flex,
                     epos,ngram,
                     e_embed_dim
                     ):
        """Creates a ShallowTranslationModel instance 

        :param model: the main model 
        :param trainer: the model optimization trainer 
        :param context_embed: the 
        :param context_embed: the context embeddings 
        :param elex: the english side lexicon 
        :param flex: the foreign side lexicon
        """
        ## dynet module
        self.dy = dynet

        self.model = model
        self.context_embed = context_embed
        self.e_embed = e_embed
        self.len_embed  = len_embed
        self.flen_embed = flen_embed
        
        ## layer stuff
        self.hlayer_list = hlist
        self.hbias_list = hbias
        self.olayer = olayer
        self.obias  = obias

        ## model updater and trainer
        self.trainer = trainer

        ## model lexicons 
        self.elex = elex
        self.flex = flex

        ## 
        self.epos = epos
        self.ngram = ngram
        self.e_embed_dim = e_embed_dim

    @classmethod
    def from_config(cls,config):
        """Build a neural shallow translator from configuration 

        :param config: the configuration 
        :returns: a translator instance 
        """
        ## laod the dependencies 
        deps =  cls._load_dynet(config)
        dy = deps[0]
        model = deps[1] 

        ## read data to get vocabulary sizes
        data = load_aligner_data(config)
        flex = data[2]
        elex = data[3]
        flen = len(flex)
        elen = len(elex)

        ## compute the input size
        ngram_size = (config.ngram-1)*config.embed_dim
        esize = config.epos*config.e_embed_dim
        input_size = ngram_size+esize+config.flen_embed_dim+config.elen_embed_dim

        ## if hidden dimension is not set, will take avarege of input and output 
        if config.hidden_dim == -1:
            config.hidden_dim = (input_size+flen)/2

            ## chech that it isn't more than double the input layer
            if config.hidden_dim >= input_size*2:
                config.hidden_dim = int(float(config.hidden_dim)*.95)

        hidden_list = []
        hidden_bias = [] 
        ## output layer
        for i in range(config.hidden_layers+1):
            if i == 0: 
                hidden_list.append(model.add_parameters((config.hidden_dim,input_size)))
                hidden_bias.append(model.add_parameters(config.hidden_dim))
            else:
                hidden_list.append(model.add_parameters((config.hidden_dim,config.hidden_dim)))
                hidden_bias.append(model.add_parameters(config.hidden_dim))
                                
        W_hy_p = model.add_parameters((flen,config.hidden_dim))
        b_y_p = model.add_parameters(flen)

        ## the e and f embeddings parameters 
        embeddings = model.add_lookup_parameters((flen,config.embed_dim))
        e_embeddings = model.add_lookup_parameters((elen,config.e_embed_dim))
        
        ## english side size embeddings
        len_embeddings = model.add_lookup_parameters((config.amax,50))
        
        ## semantic side size embeddings
        flen_embeddings = model.add_lookup_parameters((config.amax,50))

        ## trainer 
        trainer = __trainer_finder(dy,model,config)

        instance = cls(dy,model,
                           trainer,
                           embeddings,
                           e_embeddings,
                           len_embeddings,
                           flen_embeddings,
                           hidden_list,hidden_bias,
                           W_hy_p,b_y_p,
                           elex,flex,
                           config.epos,config.ngram,
                           config.e_embed_dim
                           )

        ## log details of the network 
        instance.logger.info('loaded network, with input size=%d, hidden size=%d, embedding=%d, e_embeddings=%d, output size=%d' %\
                                 (input_size,config.hidden_dim,config.embed_dim,config.e_embed_dim,flen))

        return instance

    ## training methods
    cpdef void train(self,config):
        """Train the neural network model using configuration

        :param config: the configuration with pointers to data, other info,..
        :returns: None 
        """
        data = load_aligner_data(config)

        ## try to run the trainer
        try:
            self._train(data[0],data[1],config)
        except Exception,e:
            self.logger.info(e,exc_info=True)

    cdef int _train(self,np.ndarray f,np.ndarray e,object config) except -1:
        """The main c level method for training the neural translation models 

        :param f: the foreign, or target,  data 
        :param e: the english, or source,  data
        :param config: the configuration 
        """
        cdef int w,n,i,elen,flen,size = f.shape[0]

        ## configuration stuff
        cdef int num_hidden = config.hidden_layers
        cdef int epos = self.epos
        cdef int ngram = self.ngram
        cdef int e_embed_dim = self.e_embed_dim
        cdef int elen_embed = config.elen_embed_dim
        cdef int epochs = config.epochs
        
        ## dynet stuff
        cdef object dy = self.dy
        cdef object trainer = self.trainer
        cdef double epoch_loss

        ## english and foreign sequences
        cdef int[:] eside,fside
        #cdef double[:] e_vector
        ##
        cdef list einput,elist,flist
        cdef int[:] indices = np.array(range(size),dtype=np.int32)
        cdef int index

        np.random.seed(42)
        ## log the start
        self.logger.info('Starting the training loop....')

        ## prebuild the english vector representations
        x_reps = np.zeros((size,(epos*e_embed_dim)+elen_embed,),dtype='d')
        for i in range(size):
            eside = e[i]
            x_reps[i] = self.e_rep(eside)

        self.logger.info('Pre-Built the input representations...')

        ## start new iterations
        for epoch in range(epochs):

            ## starting time 
            stime = time.time()

            ## epoch loss
            epoch_loss = 0.

            ## 0 the order 
            np.random.shuffle(indices)
                        
            ## go through data
            for i in range(size):

                index = indices[i]
                fside = f[index]
                flen = fside.shape[0]
                ## gets the representation for the input
                e_vector = dy.inputVector(x_reps[index])

                for w in range(1,flen):
                    if fside[w] == -1: continue

                    ## reset computation graph
                    dy.renew_cg()
                    ## create an expression for the input 
                    #erep = dy.inputVector(e_vector)
                    erep = dy.reuseExpression(e_vector)
                    elist = []

                    ## find the word context embeddings
                    for n in range(1,ngram):
                        if w - n >= 0 and fside[w-n] != -1:
                            elist.append(dy.lookup(self.context_embed,fside[w-n]))
                        else:
                            elist.append(dy.lookup(self.context_embed,0))

                    elist.append(erep)
                    ## current flengh embeddings
                    elist.append(dy.lookup(self.flen_embed,w))
                    ## the final representation 
                    rep = dy.concatenate(elist)

                    ## weight parameters (assumes a single layer, should be generalized)
                    hidden  = dy.parameter(self.hlayer_list[0])
                    hbias   = dy.parameter(self.hbias_list[0])
                    outbias = dy.parameter(self.obias)
                    olayer  = dy.parameter(self.olayer)

                    ## do the computation
                    out = olayer*(dy.tanh(hidden*rep)+hbias)+outbias
                    err = dy.pickneglogsoftmax(out,fside[w])
                    # ## do the updates
                    epoch_loss += err.scalar_value()
                    err.backward()
                    trainer.update(1.0)

            ## log the result
            trainer.update_epoch(1.0)

            self.logger.info('Finished epoch %d in %d seconds,loss=%f' %\
                                 (epoch,time.time()-stime,epoch_loss))

    cdef double[:] e_rep(self,int[:] x_bold):
        """Create a representation for each english side input

        In this case, the representation of the english side input, or x, is 
        just a concatenation of the word embeddings for each word in x_bold 
        up to some sequence limit (e.g., 30). Padding is used when the sequence 
        exceeds this size, and unknown words are assigned a the empty representation.

        We also include a representation of the length of the english input,
        in the form of a length embedding. 

        :param x_bold: the raw english sequence or vector representation 
        """
        cdef int w
        cdef object dy = self.dy
        cdef int epos = self.epos,x_len = x_bold.shape[0]
        cdef int e_embed_dim = self.e_embed_dim

        elist = []
        for w in range(epos):
            if w >= x_len or x_bold[w] == -1:
                ## this is pretty expensive 
                elist.append(dy.vecInput(e_embed_dim))
            else:
                elist.append(dy.lookup(self.e_embed,x_bold[w]))

        elist.append(dy.lookup(self.len_embed,x_len))
        eexpr = dy.concatenate(elist)
        e_vector = eexpr.npvalue()
        return e_vector

    cdef double score(self,double[:] x,int z,int[:] z_context,int z_pos) except -1:
        """Computes a translation score for an output word z given input x_bold

        In this case, the model just computes the 

        :param x: the input representation (could e.g., be already computed using self.e_rep) 
        :param z: the output word under consideration 
        :param z_context: the sequence context for z (to the right) 
        :param z_pos: the current position of z in the sequence
        :returns: negative log probability of output word z
        """
        cdef object dy = self.dy
        cdef int w,n,epos = self.epos 
        cdef list elist
        cdef double[:] e_vector
        cdef int ngram = self.ngram,e_embed_dim = self.e_embed_dim
        cdef int c_shape = z_context.shape[0]
        cdef double score

        ## reset the computation graph 
        #dy.renew_cg()

    @boundscheck(False)
    @wraparound(False)
    cdef int _rank(self,int[:] en,np.ndarray rl,int[:] sort,int gold_id) except -1:
        """Computes a rankled list for an input and a list of rank candidates 

        :param en: the english input 
        :param rl: the ranked list 
        :param sort: the datastructure that keeps track of scores
        :rtype: None 
        """
        cdef int rank_size = rl.shape[0]
        cdef object dy = self.dy 
        cdef int i,w,rlen,n,ngram = self.ngram
        cdef double[:] e_vector = self.e_rep(en)
        cdef double[:] problist = np.zeros((rank_size,),dtype='d')
        cdef int[:] rank_item
        cdef double prob
        cdef list elist

        erep = self.dy.inputVector(e_vector)

        ## enumerate items in the rank 
        for i in range(rank_size):
            rank_item = rl[i]
            rlen = rank_item.shape[0]
            prob = 0.0

            ## build the f context embeddings 
            for w in range(1,rlen):

                if rank_item[w] == -1: continue
                ## renew the graph
                dy.renew_cg()
                new_rep = dy.reuseExpression(erep)
                
                elist = []
                for n in range(1,ngram):
                    if w - n >= 0 and rank_item[w-n] != -1:
                        elist.append(dy.lookup(self.context_embed,rank_item[w-n]))
                    else:
                        elist.append(dy.lookup(self.context_embed,0))

                elist.append(new_rep)
                elist.append(dy.lookup(self.flen_embed,w))
                rep = dy.concatenate(elist)
                # ## hidden weights
                hidden  = dy.parameter(self.hlayer_list[0])
                hbias   = dy.parameter(self.hbias_list[0])
                outbias = dy.parameter(self.obias)
                olayer  = dy.parameter(self.olayer)

                ## compute the score
                out = olayer*(dy.tanh(hidden*rep)+hbias)+outbias
                #score = dy.pick(dy.softmax(out),rank_item[w])
                score = dy.softmax(out)[rank_item[w]]
                
                score.value()

                # if prob == 0.0:
                #     prob = dy.scalar_value(score)
                # else:
                #     prob *= dy.scalar_value(score)

            problist[i] = prob
            binary_insert_sort(i,prob,problist,sort)

    ## backup protocol
    
    def backup(self,wdir):
        """Back up the model to file 

        :param wdir: the working directory 
        """
        model_out = os.path.join(wdir,"neural_model")
        self.model.save(model_out,[
            self.context_embed,
            self.e_embed,
            self.len_embed,
            self.flen_embed,
            self.hlayer_list[0],
            self.hbias_list[0],
            self.olayer,
            self.obias,
            ])

    @classmethod
    def load_backup(cls,config):
        """Loads a backup from file 

        :param config: the main or global configuration object 
        """
        deps = cls._load_dynet(config)
        dy = deps[0]
        model = deps[1]

        ## data information
        data = load_aligner_data(config)
        flex = data[2]
        elex = data[3]
        flen = len(flex)
        elen = len(elex)

        ## location of model
        loc = os.path.join(config.dir,"neural_model")
        ## load the previous model and all of its parameters 
        ce,eembed,len_embed,flen_embed,h1,hb1,ol,obias = model.load(loc)
        trainer = __trainer_finder(dy,model,config)

        instance = cls(dy,model,trainer,
                           ce,
                           eembed,
                           len_embed,
                           flen_embed,
                           [h1],[hb1],
                           ol,obias,
                           elex,flex,
                           config.epos,config.ngram,
                           config.e_embed_dim                           
                           )

        instance.logger.info('Loaded neural model backup')
        return instance
        
cdef class AttentionFeedForward(DynetFeedForward):
    """An attention feed forward network"""

    @classmethod
    def from_config(self,config):
        """Build an attention model from configuration 

        :param config: the main configuration 
        """
        pass

            
cdef class AttentionTranslationLM(FeedForwardTranslationLM):
    """A feed forward translation lm with a (very simple!) attention mechanism, of the type 
    first described in the first publication below, and discussed in the second in relation 
    to feed-forward models. 

    @article{bahdanau2014neural,
    title={Neural machine translation by jointly learning to align and translate},
    author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
    journal={arXiv preprint arXiv:1409.0473},
    year={2014}
    }

    @article{raffel2015feed,
    title={Feed-forward networks with attention can solve some long-term memory problems},
    author={Raffel, Colin and Ellis, Daniel PW},
    journal={arXiv preprint arXiv:1512.08756},
    year={2015}
    }

    Below is the general idea (which might be crazy, I'm new to this attention stuff, let's 
    see how it works): 

    """

    @classmethod
    def from_config(cls,config):
        """Load an attention model from configuration 

        :param config: the main configuration 
        """
        deps = cls._load_dynet(config)
        dy = deps[0]
        model = deps[1]
        attention_model = dy.Model()

        ## read data to get vocabulary sizes
        data = load_aligner_data(config)
        flex = data[2]
        elex = data[3]
        flen = len(flex)
        elen = len(elex)

        ## compute the input size
        ngram_size = (config.ngram-1)*config.embed_dim
        input_size_m = ngram_size+config.e_embed_dim+config.flen_embed_dim+config.elen_embed_dim

        ##
        if config.hidden_dim == -1:
            config.hidden_dim = (input_size_m+flen)/2

            ## chech that it isn't more than double the input layer
            if config.hidden_dim >= input_size_m*2:
                config.hidden_dim = int(float(config.hidden_dim)*.95)

        hidden_list = []
        hidden_bias = [] 
        ## output layer
        for i in range(config.hidden_layers+1):
            if i == 0: 
                hidden_list.append(model.add_parameters((config.hidden_dim,input_size_m)))
                hidden_bias.append(model.add_parameters(config.hidden_dim))
            else:
                hidden_list.append(model.add_parameters((config.hidden_dim,config.hidden_dim)))
                hidden_bias.append(model.add_parameters(config.hidden_dim))
                                
        W_hy_p = model.add_parameters((2,config.hidden_dim))
        b_y_p = model.add_parameters(2)

        ## the e and f embeddings parameters 
        embeddings = model.add_lookup_parameters((flen,config.embed_dim))

        ## english side size embeddings
        len_embeddings = model.add_lookup_parameters((config.amax,50))
        
        ## semantic side size embeddings
        flen_embeddings = model.add_lookup_parameters((config.amax,50))

        ## attention model (now holds the english word embeddings) 
        e_embeddings = attention_model.add_lookup_parameters((elen,config.e_embed_dim))
        ahidden_list = [attention_model.add_parameters((config.ahidden_dim,config.e_embed_dim))]
        ahidden_bias = [attention_model.add_parameters((config.ahidden_dim))]
        W_ha_p = attention_model.add_parameters((2,config.ahidden_dim))
        b_a_p  = attention_model.add_parameters(2)

        ## main model trainer
        trainer = __trainer_finder(dy,model,config)
        attention_trainer = __trainer_finder(dy,attention_model,config)

        # instance = cls(dy,model,
        #                    trainer,
        #                    embeddings,
        #                    e_embeddings,
        #                    len_embeddings,
        #                    flen_embeddings,
        #                    hidden_list,hidden_bias,
        #                    W_hy_p,b_y_p,
        #                    elex,flex,
        #                    config.epos,config.ngram,
        #                    config.e_embed_dim,
        #                    attention_model,
        #                    )




## encoder decoder models



        
        

## helper functions


def __find_dynet(config):
    """Tries to laod the dynet module

    :raises: ImportError 
    :returns: the dyner module object 
    """
    description = """ERROR LOADING THE DYNET LIBRARY! (traceback below). To load the library, please follow the details provided in https://github.com/clab/dynet and check that the following environmental variables are included: DYNET (location of dynet) and EIGEN (path to eigen library). The location of these dependencies can also be specified when calling setup.py (see this for mor details) 
    
"""
    try:
        from zubr.cy_dynet import _dynet as dy
    except Exception,e:
        print >>sys.stderr,description
        traceback.print_exc(file=sys.stderr)
        sys.exit('\nExiting....')

    ## setup the dynet parameters from config 
    dyparams = dy.DynetParams()
    dyparams.set_mem(config.mem)
    dyparams.set_random_seed(config.seed)
    dyparams.init()
    
    return dy

def __trainer_finder(dy,model,config):
    """Find the specified trainer, to add new trainer, simply 
    put new items into the trainers dictionary

    :param dy: the dynet module 
    :param model: the neural network model 
    :param config: the global configuration
    :returns: a trainer instance 
    :raises: ValueError 
    """
    trainers = {
            "adagrad"  : dy.AdagradTrainer,
            "sgd"      : dy.SimpleSGDTrainer,
            "momentum" : dy.MomentumSGDTrainer,
            "adadelta" : dy.AdadeltaTrainer,
    }

    strainer = config.trainer.lower()
    tclass = trainers.get(strainer,None)
    
    ## check that trainer is known  
    if tclass is None:
        raise ValueError('Uknown dynet model type: %s' % strainer)

    ## init each trainer according to required properties
    if strainer   == "adagrad":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay,eps=config.epsilon)
    elif strainer == "sgd":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay)
    elif strainer == "momentum":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay,mom=config.momentum)
    elif strainer == "adadelta":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay,rho=config.rho)

    return trainer


NMs = {
    "trans_lm" : FeedForwardTranslationLM,
}


def NeuralLearner(model_type):
    """Factory method for retrieving a neural network class 

    :param model_type: the type of model desired 
    :raises: ValueError 
    """
    neural_class = NMs.get(model_type,None)

    if neural_class is None:
        raise ValueError('Unknown neural model type: %s' % model_type)
    return neural_class

    
def params():
    """Parameters for the neural models module 

    :rtype: tuple 
    :returns: description of options with names, and list of options 
    """
    from zubr.Alignment import params as aparams
    agroup,aparams = aparams()

    options = [
        ("--ngram","ngram",3,"int",
         "The type of ngram model to use [default=2]","NeuralLM"),
        ("--hidden_dim","hidden_dim",-1,"int",
         "The size or dimension of the hidden layer [default=-1]","NeuralModels"),
        ("--activation","activation","tanh","str",
         "The non-linear activation function to use [default=tanh]","NeuralModels"),
        ("--weight_decay","weight_decay",0.0,"float",
         "Learning rate decay [default=0.0]","NeuralModels"),
        ("--lrate","lrate",0.1,"float",
         "The initial learing rate [default=0.1]","NeuralModels"),
        ("--epochs","epochs",5,int,
         "The number of iterations to run on data [default=200]","NeuralLM"),
        ("--embed_dim","embed_dim",200,int,
         "The size of the context word embeddings [default=200]","NeuralLM"),
        ("--epsilon","epsilon",1e-20,float,
         "Epsilon parameter to prevent numerical instability [default=le-20]","NeuralModels"),
        ("--momentum","momentum",0.9,float,
         "Momemtum value (for momentum models) [default=le-20]","NeuralModels"),
        ("--rho","rho",0.95,float,
         "Update parameter for moving average of updates [default=0.95]","NeuralModels"),
        ("--trainer","trainer","sgd","str",
         "The type of trainer to use [default='sgd']","NeuralModels"),
        ("--epos","epos",30,int,
         "The maximum number of vocabulary items [default=30]","NeuralTrans"),
        ("--e_embed_dim","e_embed_dim",50,int,
         "The size of each english side embedding [default=50]","NeuralTrans"),
        ("--elen_embed_dim","elen_embed_dim",50,int,
         "The size of the lengnth embedding[default=50]","NeuralTrans"),
        ("--flen_embed_dim","flen_embed_dim",50,int,
         "The size of the lengnth embedding [default=50]","NeuralTrans"),
        ("--hidden_layers","hidden_layers",1,int,
         "The number of hidden layers [default=1]","NeuralTrans"),
        ("--nmodel","nmodel","trans_lm","str",
         "The type of neural network architecture to use [default=trans_lm]","NeuralModels"),
        ("--seed","seed",3249411197 ,float,
         "Random seed for dynet [default=3249411197]","NeuralModels"),
        ("--mem","mem",512,int,
         "The amount of memory to allocate to dynet [default=512mb]","NeuralModels"),
        ("--ahidden_dim","ahidden_dim",512,int,
         "The size of the attention hidden layer [default=500]","NeuralModels"),
        # ("--len_feat","lean_feat",False,bool,
        #  "Include features about input/output lengths [default=False]","NeuralTrans"),
        ]
        
    options += aparams
    group = {
        "NeuralModels" : "general settings for neural models",
        "NeuralLM"     : "settings for neural language models and related",
        "NeuralTrans"  : "settings for neural translation models and related",
    }
        
    return (group,options)

def argparser():
    """Returns an argument parser for the neural models module"""
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj

    usage = """python -m zubr neural [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

def main(argv):
    """Main execution point for running neural models from zubr top-level or in a pipeline

    :param argv: cli arguments or a condiguration 
    :type argv: list or zubr.util.ConfigAttrs
    """
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])
        logging.basicConfig(level=logging.DEBUG)

    ## main execution point
    try:
         nclass = NeuralLearner(config.nmodel)

         ## load from configuration
         model = nclass.from_config(config)

         ## train the model
         model.train(config)
        
    except Exception,e:
        traceback.print_exc(file=sys.stdout)
