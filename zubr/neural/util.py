import os
import shutil
import codecs
import logging
import random
import subprocess
import time
import datetime 
import numpy as np
from collections import defaultdict
from random import choice,randrange,seed
from zubr.util.alignment_util import load_aligner_data
from zubr.util.aligner_util import get_rdata
from zubr.Alignment import params as aparams

from zubr.neural.Seq2Seq import (
    ParallelDataset,
    SymbolTable,
    TransitionTable,
    NeuralModel,
    RerankList,
)
    


__all__ = [
    "build_data",
    "pad_input",
    "oov_graph",
    "build_constraints",
    "backup_model",
    "load_best",
    "sanity_check",
    "reranker_data",
    "load_new",
    "time_backup",
]

util_logger = logging.getLogger("zubr.neural.util")
EOS = "*end*"


def sanity_check():
    """Sanity check that the lex models and neural models have the same lexicons, etc.."""
    pass


def reranker_data(config,eos):
    """Add a reranker dataset if desired

    :param config: the global configuration 
    """
    rerank = os.path.join(config.wdir,"rerank")
    encoded = os.path.join(config.dir,"encoded_rank_list.txt")
    valid_ranks = os.path.join(rerank,"valid_ranks.txt")

    ## check that the data is available 
    if not os.path.isdir(rerank) or not os.path.isfile(valid_ranks) or not os.path.isfile(encoded):
        util_logger.warning('No reranker data found, skipping!!!')
        return RerankList.build_empty()

    elif not config.rerank:
        return RerankList.build_empty()

    reps = {}
    sentence_ranks  = {}
    sentence_gold   = {}
    sentence_scores = set()

    util_logger.info('Exracting representations for reranker component...')
    ## get the representations 
    with open(encoded) as my_encoded:
        for line in my_encoded:
            line = line.strip()
            sem_id,_,vector = line.split('\t')
            sem_id = int(sem_id)
            vector = np.array([0 if i == '-1' else int(i) for i in vector.split()+[eos]],dtype=np.int32)
            vector[0] = eos
            reps[sem_id] = vector

    ## look at reranker stuff
    total = 0
    sentence_ranks = []    
            
    with open(valid_ranks) as my_ranks:
        for k,line in enumerate(my_ranks):
            total += 1
            line = line.strip()
            sentence_id,gold_id,prediction_list = line.split('\t')
            sentence_id = int(sentence_id)
            gold_id = int(gold_id)
            sentence_gold[sentence_id] = gold_id
            prediction_list = [int(p) for p in prediction_list.split()]
            if prediction_list[0] == gold_id: sentence_scores.add(sentence_id)
            ## make sure that the ordering is correct 
            assert k == sentence_id,"ordering wrong"
            ## top items 
            top   = [p for p in prediction_list[:15] if p != gold_id]
            final = [gold_id]+top[:9]
            sentence_ranks.append(final)

    ## select portion
    sentence_ranks = np.array(sentence_ranks,dtype=np.int32)
    return RerankList(sentence_ranks,reps,sentence_scores)
            
            
def backup_model(model,wdir,epoch):
    """Back up a model during training 

    :param model: the model to backup 
    :param wdir: the current working directory 
    :rtype: None
    """
    ## do nothign if there is not working directory specified 
    if wdir is None:
        util_logger.warning('No working directory specified, not backing up...')
        return

    ## output directory 
    best_model = os.path.join(wdir,"best_models")
    ## remove the current best 
    if os.path.isdir(best_model): shutil.rmtree(best_model)
    os.mkdir(best_model)
    util_logger.info('backing up best model after epoch: %s' % str(epoch+1))
    model.backup(best_model)

def time_backup(wdir,model):
    bt = time.time()
    name = os.path.join(wdir,datetime.datetime.fromtimestamp(bt).strftime('backup_%Y-%m-%d-%H:%M:%S'))
    os.mkdir(name)
    util_logger.info('backing up model after long stretch....')
    model.backup(name)
    

def load_best(config,best_e):
    """Load the best model from an experiment run 

    :param config: the global configuration 
    :returns: the best model backed up during training 
    """
    util_logger.info('Loading best model from epoch=%d' % best_e)
    best_model = os.path.join(config.dir,"best_models")
    
    ## change wdir for loading modeling 
    odir = config.dir
    config.dir = best_model
    nclass = NeuralModel(config.model)
    model = nclass.load_backup(config)
    
    ## restore old wdir
    config.dir = odir
    shutil.rmtree(best_model)
    return model

def load_new(config):
    """Load a new model 

    :param config: the global experiment configuration 
    :returns: a loaded neural model 
    """
    if not config.from_neural or not os.path.isdir(config.from_neural):
        raise ValueError('Cannot find the model: %s' % str(config.from_neural))

    
    odir = config.dir
    config.dir = config.from_neural
    nclass = NeuralModel(config.model)
    model = nclass.load_backup(config)
    config.dir = odir
    return model        

def build_constraints(nmodel,train_data,dend):
    """Build transition constraints from training data

    :param nmodel: the type of neural network being used
    :param train_data: the training data 
    :param dend: 
    """
    if nmodel != 'cattention':
        return None 

    print nmodel
    exit('exited prematurely...')

def oov_graph(labels):
    """Replace -1 in the graph with 0 

    :param labels: the graph labels 
    """
    for k,label in enumerate(labels):
        if label == -1:
            labels[k] = 0

def pad_input(dinput,eos):
    """Pad the data input with <EOS> at beginning and end

    :param dinput: the input to add 
    :rtype: None 
    """
    for k,sequence in enumerate(dinput):
        new_seq = np.insert(np.insert(sequence,0,eos),sequence.shape[0]+1,eos)
        for w,word in enumerate(new_seq):
            if word == -1: new_seq[w] = 0
        dinput[k] = new_seq

def __sample_model(min_length, max_lenth,characters):
    random_length = randrange(min_length, max_lenth)
    random_char_list = [choice(characters[:-1]) for _ in range(random_length)]
    random_string = ''.join(random_char_list) 
    return random_string, random_string[::-1]

def __build_data(config):
    """Build general datasets 

    """
    data = load_aligner_data(config)
    flex = data[2]
    elex = data[3]

    ## add <eos> to each item
    flex[EOS] = len(flex)
    elex[EOS] = len(elex)

    ## put in <EOS> and </EOS> into both representations
    foreign = data[0]
    english = data[1]

    assert flex[EOS] == (len(flex)-1),"wrong eos foreign"
    assert elex[EOS] == (len(elex)-1),"wrong eos english"
    util_logger.info('foreign side <EOS> located at: %s' % str(len(flex)-1))
    util_logger.info('english side <EOS> located at: %s' % str(len(elex)-1))
    dend = len(flex)-1
    
    ## ADD <EOS> representations to each
    word_counts = defaultdict(int)
    ecounts = defaultdict(int)
    
    ## foreign sequences 
    for k,sequence in enumerate(foreign):
        # new_seq = np.insert(np.insert(sequence[1:],0,flex["<EOS>"]),
        #                         sequence.shape[0],flex["<EOS>"])
        new_seq = np.insert(np.insert(sequence[1:],0,flex[EOS]),
                                sequence.shape[0],flex[EOS])
        foreign[k] = new_seq
        for word in new_seq:
            word_counts[word] += 1

    # ## english sequences
    for k,sequence in enumerate(english):
        # new_seq = np.insert(np.insert(sequence,0,elex["<EOS>"]),
        #                         sequence.shape[0]+1,elex["<EOS>"])
        new_seq = np.insert(np.insert(sequence,0,elex[EOS]),
                                sequence.shape[0]+1,elex[EOS])
        english[k] = new_seq
        for word in sequence:
            ecounts[word] += 1

    # ## oov ? use 0 to denote the <unk> token  
    if config.add_oov:
        util_logger.info('Added OOV examples in training...')
        added_e = 0; added_f = 0
        total_e = 0; total_f = 0

        ## first through foreign
        for k,sequence in enumerate(foreign):
            for w,word in enumerate(sequence):
                total_f += 1
                if word_counts[word] <= config.oov_threshold and added_f <= config.max_oov:
                    added_f += 1
                    sequence[w] = 0

        ## then through english
        for k,sequence in enumerate(english):
            for w,word in enumerate(sequence):
                total_e += 1
                if ecounts[word] <= config.oov_threshold and added_e <= config.max_oov:
                    added_e += 1
                    sequence[w] = 0

        util_logger.info('Finished adding OOV: e=%d (/%d),f=%d (/%d)' % (added_e,total_e,added_f,total_f))

    ## training data 
    train_data = ParallelDataset(english,foreign)
    table = SymbolTable(elex,flex)

    ## try to build validation 
    try: 
        rl,inp,order,freq,enorig = get_rdata(config,flex,elex,ttype='valid')
        valid_e = inp[0]
        assert inp[0].shape[0] == inp[1].shape[0],"data mismatch"

        ## update and add oov to validation 
        for k,sequence in enumerate(valid_e):
            new_seq = np.insert(np.insert(sequence,0,elex[EOS]),sequence.shape[0]+1,elex[EOS])
            for w,word in enumerate(new_seq):
                if word == -1: new_seq[w] = 0
            valid_e[k] = new_seq

        ## create the foreign list
        output = []
        for k,item in enumerate(inp[1]):
            ## get rid of zeros 
            rep = [flex.get(i,0) for i in ("*end* %s *end*" % order[item]).split()]
            output.append(np.array(rep,dtype=np.int32))
        valid_data = ParallelDataset(valid_e,np.array(output,dtype=object))

    except Exception,e:
        util_logger.warning('Error making validation data... creating empty')
        util_logger.error(e,exc_info=True)
        valid_data = ParallelDataset.make_empty()

    ## find transition constraints
    #############################
    
    next_item = [set() for i in flex]
    ## make sure every word can come after OOV word = 0
    next_item[0] = set(flex.values())

    ## go through rank list 
    for rank_item in rl:
        seq = [dend]+[i if i != -1 else 0 for i in rank_item[1:]]+[dend]
        for k,item in enumerate(seq):
            if k == len(seq)-1: continue
            next_item[item].add(seq[k+1])

    # ## go through the training data just in case
    for seq in foreign:
        for k,item in enumerate(seq):
            if k == len(seq)-1: continue
            next_item[item].add(seq[k+1])


    ## make the final list
    transitions = []
    for item in next_item:
        sorted_item = np.array(sorted(item),dtype=np.int32)
        lookup = {i:k for k,i in enumerate(sorted_item)}
        transitions.append((sorted_item,lookup))
        
    assert len(transitions) == len(flex),"wrong transitions"
    trans_obj = TransitionTable(transitions)
    
    ## log the dataset sizes
    util_logger.info('Train data contains %d examples' % train_data.size)
    util_logger.info('Validation data contains %d examples' % valid_data.size)

    #return (train_data,valid_data,table)
    return (train_data,valid_data,table,trans_obj)


def __build_demo_data(config):
    """Build demo data for running the seq2seq modela

    :param config: the global configuration 
    """
    seed(a=42)
    characters = list("abcd")
    #characters.append("<EOS>")
    characters.append(EOS)

    int2char = list(characters) 
    char2int = {c:i for i,c in enumerate(characters)}

    train_set = [__sample_model(1,config.max_string,characters) for _ in range(3000)]
    val_set   = [__sample_model(1,config.max_string,characters) for _ in range(50)]

    ## training data 
    source,target = zip(*train_set)
    source =  np.array([np.array([char2int[c] for c in list(i)+[EOS]],dtype=np.int32) for i in source],dtype=object)
    target =  np.array([np.array([char2int[c] for c in list(i)+[EOS]],dtype=np.int32) for i in target],dtype=object)
    train_data = ParallelDataset(source,target)

    ## valid data
    sourcev,targetv = zip(*val_set)
    sourcev = np.array([np.array([char2int[c] for c in list(i)+[EOS]],dtype=np.int32) for i in sourcev],dtype=object)
    targetv =  np.array([np.array([char2int[c] for c in list(i)+[EOS]],dtype=np.int32) for i in targetv],dtype=object)
    valid_data = ParallelDataset(sourcev,targetv)

    ## symbol table
    table = SymbolTable(char2int,char2int)

    ## print the data
    if config.dir: 
        source = os.path.join(config.dir,"data.e")
        target = os.path.join(config.dir,"data.f")
        source_eval = os.path.join(config.dir,"data_val.e")
        target_eval = os.path.join(config.dir,"data_val.f")
        rfile = os.path.join(config.dir,"rank_list.txt")
        in_rank = set()

        with codecs.open(rfile,'w',encoding='utf-8') as ranks: 
            ## train data 
            with codecs.open(source,'w',encoding='utf-8') as sourcet:
                with codecs.open(target,'w',encoding='utf-8') as targett:
                    for (dinput,output) in train_set:
                        print >>sourcet,' '.join(dinput)
                        print >>targett,' '.join(output)
                        if output not in in_rank:
                            print >>ranks,' '.join(output)
                        in_rank.add(output)
                        if dinput not in in_rank:
                            print >>ranks,' '.join(dinput)
                        in_rank.add(dinput)

            ## eval data
            with codecs.open(source_eval,'w',encoding='utf-8') as source_v:
                with codecs.open(target_eval,'w',encoding='utf-8') as target_v:
                    for (input,output) in val_set:
                        print >>source_v,' '.join(input)
                        print >>target_v,' '.join(output)
                        if output not in in_rank: 
                             print >>ranks,' '.join(output)
                        in_rank.add(output)
                        if input not in in_rank:
                            print >>ranks,' '.join(input)
                        in_rank.add(input)
                        
                        last_car = output[-1]
                        moutput = output+last_car


    return (train_data,valid_data,table,None)

def __build_demo_data2(config):
    """Data for running the second demo"""
    #eos = "<EOS>"
    eos = EOS
    characters = list("abcdefghijklmnopqrstuvwxyz ")
    characters.append(eos)
    char2int = {c:i for i,c in enumerate(characters)}
    train_set = [("it is working","it is working")]
    
    ## training data
    source,target = zip(*train_set)
    source = np.array([np.array([char2int[w] for w in [EOS]+list(i)+[EOS]],dtype=np.int32) for i in source])
    target = np.array([np.array([char2int[w] for w in [EOS]+list(i)+[EOS]],dtype=np.int32) for i in target])

    
    train_data = ParallelDataset(source,target)
    table = SymbolTable(char2int,char2int)
    valid = ParallelDataset.make_empty()
    return (train_data,valid,table,None)

def build_data(config):
    """General method to build data for running Seq2Seq models

    :param config: the global configuration 
    :rtype: tuple 
    """
    if config.demo_data:
        return __build_demo_data(config)
    elif config.demo_data2:
        return __build_demo_data2(config)
    else:
        return __build_data(config) 


def params():
    """Utility and data parameters for running neural models"""
    options = [
        ("--demo_data","demo_data",False,"bool",
         "Run some data data for seq2seq model [default=False]","NeuralUtil"),
        ("--max_string","max_string",15,int,
         "The maximum string (for demo) [default=15]","NeuralUtil"),
        ("--demo_data2","demo_data2",False,"bool",
         "Run the demo data again [default=False]","NeuralUtil"),
        ("--add_oov","add_oov",True,"bool",
         "Add unknown word tokens to input [default=True]","NeuralUtil"),
        ("--max_oov","max_oov",200,int,
         "The maximum number of oov words to replace [default=100]","NeuralUtil"),
        ("--oov_threshold","oov_threshold",2,int,
         "The threshold  [default=2]","NeuralUtil"),
        ("--rfile","rfile","","str",
         "Path to the rank file [default=""]","NeuralUtil"),
    ]

    group = {"NeuralUtil": "Utility functions for running neural models"}
    agroup,aoptions = aparams()
    group.update(agroup)
    options += aoptions
    return (group,options) 
