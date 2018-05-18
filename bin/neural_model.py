import os
import re
import shutil
import logging
import codecs
import subprocess

params = [
    ("--wdir","wdir",'',"str",
     "The working directory [default='']","NeuralRunner"),
    ("--name","name",'',"str",
     "The name of the dataset [default='']","NeuralRunner"),
    ("--bow","bow",False,"bool",
     "Use the smaller _bow training data [default=False]","NeuralRunner"),
    ("--make_subword","make_subword",False,"bool",
     "Use the subword representations for english side [default=False]","NeuralRunner"),
    ("--make_sem_subword","make_sem_subword",False,"bool",
     "Use the subword representations for english side [default=False]","NeuralRunner"),     
    ("--add_extra","add_extra",False,"bool",
     "Add extra data to the neural parallel training data [default=False]","NeuralRunner"),
    ("--num_symbols","num_symbols",1000,"int",
     "The number of symbols for the subword builder [default=False]","NeuralRunner"),
    ("--trans","trans",False,"bool",
     "Use transliterated version of data (if available) [default=False]","NeuralRunner"),
    ("--mixed","mixed",False,"bool",
     "Use mixed data (if available) [default=False]","NeuralRunner"),
    ("--add_hiero","add_hiero",False,"bool",
     "Add hiero rules to training (if available) [default=False]","NeuralRunner"),
    ("--double_data","double_data",False,"bool",
     "Double the data [default=False]","NeuralRunner"),     
]

    
description = {"NeuralRunner" : "settings for running neural model"}

tasks = [
    "setup_data",
    "zubr.wrapper.foma",
    "link_graph",
    "zubr.neural.run"
]

script_logger = logging.getLogger("neural_model")

def link_graph(config):
    if not config.graph: 
        config.graph = os.path.join(config.dir,"graph.att")

def __subword(config):
    from zubr.util.apply_bpe import from_data
    from_data(config)

def __double_data(config,e,f):
    #c1 = subprocess.Popen('cat %s %s > %s' % (e,e,e))
    os.system("cat %s %s > %s" % (e,e,os.path.join(config.dir,"new.e")))
    os.system("cat %s %s > %s" % (f,f,os.path.join(config.dir,"new.f")))
    shutil.copy(os.path.join(config.dir,"new.e"),e)
    shutil.copy(os.path.join(config.dir,"new.f"),f)
    # c1.wait()
    # c2 = subprocess.Popen('cat %s %s > %s' % (f,f,f))
    # c2.wait()
    
def __extra_pairs(pseudo,extra,new_e,new_f):
    """Extract single occurrences of the extra data to use for training data

    :param pseudo: the pseudo lexicon entries 
    :param extra: the extra parallel pairs 
    """
    pairs = set()
    
    if os.path.isfile(pseudo):
        with codecs.open(pseudo,encoding='utf-8') as lex:
            for line in lex:
                line = line.strip()
                try: 
                    w1,w2 = line.split('\t')
                    w1 = w1.strip().lower()
                    w2 = w2.strip().lower()
                    ## don't add too much 
                    if w1 != w2: continue
                    pairs.add((w2,w1))
                except: pass 
    if extra:
        with codecs.open(extra,encoding='utf-8') as more:
            for line in more:
                line = line.strip()
                sem,en = line.split('\t')
                sem = sem.strip().lower()
                en = en.strip().lower()
                pairs.add((en,sem))

    ## add to the training data 
    if pairs:
        with codecs.open(new_e,'a',encoding='utf-8') as english:
            with codecs.open(new_f,'a',encoding='utf-8') as foreign:
                for (en,sem) in pairs:
                    print >>english,en.strip()
                    print >>foreign,sem.strip()

def __add_hiero_data(hiero_path,new_e,new_f):
    """Add parallel hiero data to the training """
    toadd = set()
    with codecs.open(hiero_path) as my_hiero:
        for line in my_hiero:
            line = line.strip()
            lhs,rhs,freq = line.split('\t')
            english_raw,sem_raw = rhs.split(' ||| ')
            freq = int(freq.split('=')[-1])

            ## without the
            english = re.sub(r'\s+',' ',re.sub(r'\[.+\]',' ',english_raw).strip())
            sem = re.sub(r'\s+',' ',re.sub(r'\[.+\]',' ',sem_raw).strip())

            ## SKIP 
            if english and sem:
                toadd.add((english,sem))

            ## the cases with gaps
            if (english != english_raw or sem != sem_raw) and english and sem:
                toadd.add((english_raw,sem_raw))

    script_logger.info('Number of hiero rules added: %d' % len(toadd))
    ## add to training data
    with codecs.open(new_e,'a',encoding='utf-8') as english:
            with codecs.open(new_f,'a',encoding='utf-8') as foreign:
                for (en,sem) in toadd:
                    print >>english,en.strip()
                    print >>foreign,sem.strip()
                    
def setup_data(config):
    """Make sure data directory exists, ...

    :param config: the global configuration 
    :rtype: None
    """
    ## demo data
    if config.demo_data:
        from zubr.neural.util import build_data
        config.name = 'data'
        build_data(config)
        config.rfile = os.path.join(config.dir,"rank_list.txt")
        config.atraining = os.path.join(config.dir,"data")
        config.demo_data = False
        return

    ## check that working directory is specified and exists 
    if config.wdir is None:
        exit('Please specify data directory! exiting...')
    if not os.path.isdir(config.wdir):
        exit('Cannot find the specified working directory! exiting...')
    if config.name is None:
        exit('Please specify the name of the data! exiting...')

    ## check for the training data
    name_base = os.path.join(config.wdir,config.name)
    train_name = name_base if (not config.bow and not config.add_extra) else name_base+"_bow"

    ### make version of data for alignment model
    ab_full_e = name_base+"."+config.target
    ab_full_f = name_base+"."+config.source

    ## for the lexical model inside neural model (if used) 
    shutil.copy(ab_full_e,os.path.join(config.dir,config.name+"_lex."+config.target))
    shutil.copy(ab_full_f,os.path.join(config.dir,config.name+"_lex."+config.source))

    ## setup the training data
    orig_e = train_name+"."+config.target
    orig_f = train_name+"."+config.source
    orig_e_trans = train_name+"_trans."+config.target
    new_e = os.path.join(config.dir,"%s.%s" % (config.name,config.target))
    new_f = os.path.join(config.dir,"%s.%s" % (config.name,config.source))

    ## check the data existence 
    if not os.path.isfile(orig_e):
        exit('Cannot find the specified source file! exiting...')
    if not os.path.isfile(orig_f):
        exit('Cannot find the specified target file! exiting...')

    ## copy over training data
    if config.trans:
        shutil.copy(orig_e_trans,new_e)
    else:
        shutil.copy(orig_e,new_e)
    shutil.copy(orig_f,new_f)
    config.atraining = os.path.join(config.dir,config.name)

    if config.add_extra:
        e1 = os.path.join(config.wdir,"extra_pairs.txt")
        e2 = os.path.join(config.wdir,"__extra_pairs.txt")
        pseudo = os.path.join(config.wdir,"pseudolex.txt")

        ## check if there are extra examples 
        if os.path.isfile(e1) or os.path.isfile(e2):
            extra = e1 if os.path.isfile(e1) else e2
        else:
            extra = None

        ## add it to the training data
        __extra_pairs(pseudo,extra,new_e,new_f)

    hiero = os.path.join(config.wdir,"hiero_rules.txt")
    if config.add_hiero and os.path.isfile(hiero):
        __add_hiero_data(hiero,new_e,new_f)

    elif config.add_hiero:
        script_logger.error('Cannot find hiero rules! Skipping...')

    ## copy the rank file
    rank_file = os.path.join(config.wdir,"rank_list.txt")
    if not os.path.isfile(rank_file):
        exit('Cannot find the rank file! exiting...')
    shutil.copy(rank_file,config.dir)
    config.rfile = os.path.join(config.dir,"rank_list.txt")

    ## copy bow data if it exists
    try:
        shutil.copy(os.path.join(config.wdir,"%s_bow.%s" % (config.name,config.target)),config.dir)
        shutil.copy(os.path.join(config.wdir,"%s_bow.%s" % (config.name,config.source)),config.dir)
    except Exception:
        script_logger.info('Did not find _bow (ignore if not used)...')

    ## double the data
    if config.double_data:
        __double_data(config,new_e,new_f)
    
    ## validation data
    try:
        orig_ev = name_base+"_val."+config.target
        orig_ev_trans = name_base+"_val_trans."+config.target
        orig_fv = name_base+"_val."+config.source
        mixed_e = name_base+"_val_mixed."+config.target
        mixed_f = name_base+"_val_mixed."+config.source

        if config.trans:
            shutil.copy(orig_ev_trans,os.path.join(config.dir,"%s_val.%s" % (config.name,config.target)))
        elif config.mixed:
            script_logger.info('Using mixed dev set...')
            shutil.copy(mixed_e,os.path.join(config.dir,"%s_val.%s" % (config.name,config.target)))
        else:
            shutil.copy(orig_ev,config.dir)

        if config.mixed:
            shutil.copy(mixed_f,os.path.join(config.dir,"%s_val.%s" % (config.name,config.source)))
            script_logger.info('Using mixed dev set (f side)...')
        else:
            shutil.copy(orig_fv,config.dir)

        
        v_lang = name_base+"_val.language"
        try:
            shutil.copy(v_lang,config.dir)
        except:
            script_logger.info('Note: did not find valid language names (if not polyglot, ignore)')
    except Exception,e:
        script_logger.error(e,exc_info=True)
        #script_logger.warning('Error building/finding the validation data')

    ## test data
    try:
        orig_ef = name_base+"_test."+config.target
        orig_ef_trans = name_base+"_test_trans."+config.target
        orig_ff = name_base+"_test."+config.source
        mixed_e = name_base+"_test_mixed."+config.target
        mixed_f = name_base+"_test_mixed."+config.source
        
        if config.trans:
            shutil.copy(orig_ef_trans,os.path.join(config.dir,"%s_test.%s" % (config.name,config.target)))
        elif config.mixed:
            shutil.copy(mixed_e,os.path.join(config.dir,"%s_test.%s" % (config.name,config.target)))
            script_logger.info('Using mixed test set...')
        else: shutil.copy(orig_ef,config.dir)


        ###
        if config.mixed:
            shutil.copy(mixed_f,os.path.join(config.dir,"%s_test.%s" % (config.name,config.source)))
        else:
            shutil.copy(orig_ff,config.dir)

        t_lang = name_base+"_test.language"
        try:
            shutil.copy(t_lang,config.dir)
        except:
            script_logger.info('Note: did not find test language names (if not polyglot, ignore)')
        
    except Exception,e:
        script_logger.warning('Error building/finding the test data')


    ## creat sub-word representations?

    if config.make_subword or config.make_sem_subword: __subword(config)

    ## additional global adjustments to configuration 
    config.amax = 250
    config.extract_phrases = False 
    config.extract_hiero = False
