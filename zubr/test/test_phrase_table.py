import logging
import time
import os
import shutil
import codecs 
from zubr.Phrases import PhraseTable,argparser as table_config
from zubr import lib_loc

if __name__ == "__main__":
    config = table_config().default_config()
    config.override = True
    logging.basicConfig(level=logging.INFO)
    
    ## copy over the phrase table

    ## FIRST TEST ON ENGLISH
    ephrases    = os.path.join(lib_loc,"examples/_sandbox/english_table.txt")
    edummydir   = os.path.join(lib_loc,"examples/_sandbox/ephrase_test")
    dphrases    = os.path.join(edummydir,"phrase_table.txt")

    ## create the dummy directory 
    os.mkdir(edummydir)
    config.dir = edummydir
    shutil.copy(ephrases,dphrases)
    table1 = PhraseTable(config).from_config(config)
    t1_alist   = table1.slist
    t1_phrases = table1.phrases
    st = time.time()
    
    with codecs.open(ephrases,encoding='utf-8') as my_phrases:
        for k,line in enumerate(my_phrases):
            left,right,_ = line.split('\t')
            table_id = table1.query(left,right)
            target_str = "%s ||| %s" % (left.strip(),right.strip())
            in_table = t1_phrases[t1_alist[table_id]]
            if target_str != t1_phrases[t1_alist[table_id]]:
                print "ERROR: \n \tinput: %s \n \tother: %s" % (target_str,in_table)
                
            assert table1.query("random","shit") == -1
                
    #print time.time()-st
    shutil.rmtree(edummydir)
    print time.time()-st

    
    ## TEST ON greek
    grphrases    = os.path.join(lib_loc,"examples/_sandbox/gr_phrase_table.txt")
    edummydir   = os.path.join(lib_loc,"examples/_sandbox/gphrase_test")
    dphrases    = os.path.join(edummydir,"phrase_table.txt")
    os.mkdir(edummydir)
    config.dir = edummydir
    shutil.copy(ephrases,dphrases)
    table1 = PhraseTable(config).from_config(config)
    t1_alist   = table1.slist
    t1_phrases = table1.phrases

    with codecs.open(ephrases,encoding='utf-8') as my_phrases:
        for k,line in enumerate(my_phrases):
            left,right,_ = line.split('\t')
            table_id = table1.query(left,right)
            target_str = "%s ||| %s" % (left.strip(),right.strip())
            in_table = t1_phrases[t1_alist[table_id]]
            if target_str != t1_phrases[t1_alist[table_id]]:
                print "ERROR: \n \tinput: %s \n \tother: %s" % (target_str,in_table)
    
    shutil.rmtree(edummydir)


    ## hiero rules
    ehiero = os.path.join(lib_loc,"examples/_sandbox/hiero_rules.txt")
    edummy = os.path.join(lib_loc,"examples/_sandbox/ehiero")
    try: 
        os.mkdir(edummy)
    except:
        pass
    config.dir = edummy
    shutil.copy(ehiero,edummy)
    config.pt_type = 'hierotable'

    table3 = PhraseTable(config).from_config(config)
    t3_alist = table3.slist
    t3_phrases = table3.phrases
    
    t3 = time.time()
    with codecs.open(ehiero,encoding='utf-8') as my_hiero:
        for k,line in enumerate(my_hiero):
            line = line.strip()
            left,right,_ = line.split('\t')
            english,foreign = right.split(' ||| ')
            target = "%s ||| %s ||| %s" % (left,english,foreign)
            table_id = table3.query(english,foreign,lhs=left)
            in_table = t3_phrases[t3_alist[table_id]]

            if target != in_table:
                print "ERROR: \n \tinput: %s \n \tother: %s" % (target,in_table)

    
    print time.time()-t3
    
    
    
