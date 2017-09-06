## reimplementation of the term baseline from Deng,Chrupala LREC
## their original code is here: https://bitbucket.org/gchrupala/codeine

import sys
import numpy as np
from itertools import chain,izip
import codecs
import os 

def parse_rank_file(rank_file,enc):
    """parse the file containing all target signature representations

    :param rank_file: path to rank file
    :type rank_file: str
    :param enc: encoding to use when reading the data
    :type enc: str
    :returns: tuple of all signatures in dict form, plus list
    :rtype: tuple 
    """
    
    with codecs.open(rank_file,encoding=enc) as ranks:
        documents = {}
        original = []

        for k,outputrep in enumerate(ranks):
            outputrep = outputrep.strip()
            outputrep = outputrep.lower()
            original.append(outputrep) 
            words = outputrep.split()
            for w in words:
                Dix = documents.get(k,{})
                Dix[w] = Dix.get(w,0.0)+1 
                documents[k] = Dix

    length = list(sum(documents[i].values()) for i in documents.keys())
    length = np.array(length)
    return (documents,original,length)
    

def find_word_occurences(aname,enc):
    """compute the (relative) frequency of words in training

    -- note that all words are lowercased

    :param aname: directory where data sits
    :type aname: str
    :param enc: text encoding to use when reading file
    :type enc: str
    :returns: dictionary containing words to frequencies
    :rtype: dict
    """
    etrain = "%s.e" % aname
    counter = {}
    prob_table = {}
    
    with codecs.open(etrain,encoding=enc) as training:
        for train_sample in training:
            train_sample = train_sample.strip().lower()
            for w in train_sample.split():
                counter[w] = counter.get(w,0) + 1.0
                
    denom = sum(counter.values())
    for w,p in counter.iteritems():
        prob_table[w] = p/denom

    return prob_table
            
def parse_queries(aname,original,enc):
    """go through the queries or test examples and signatures

    -- note that everything is lowercased
    
    :param aname: name of alignment training data
    :type aname: str
    :param original: the rank list
    :type original: list
    :param enc: the encoding to use
    :type enc: str
    :returns: each test query with id of its gold representation
    :rtype: list
    """
    efile = "%s_test.e" % (aname)
    ffile = "%s_test.f" % (aname)
    pairs = []

    with codecs.open(efile,encoding=enc) as e:
        with codecs.open(ffile,encoding=enc) as f:
            en = e.readlines()
            sem = f.readlines()
            assert len(en) == len(sem),"size mismatches"
            test_len = len(en)

            for indx in range(test_len):
                sen = en[indx]
                rep = sem[indx]
                rep = rep.strip().lower()
                sen = sen.strip().lower()
                if not sen or not rep or len(sen.split()) <= 1:
                    continue 
                rep_indx = original.index(rep)
                pairs.append((sen,rep_indx))

    return pairs

## main method for the ranking 

def rank(q,indx,indices,docs,tlambda,length,probs):
    """find the rank of the highest scoring signature. This porition is lifted
    more or less directly from Deng,Chrupala (see link at top of page)

    :param q: text query
    :type q: list
    :param indices: indices of the signatures in rank list
    :type indices: list
    :param docs: the rank list in the form of a dict
    :type doc: dict
    """
    vocab = probs.keys()
    prob_words = np.array([[probs[w]] for w in q if w in vocab])
    freq_match = np.array([[(docs[d].get(w,0) / sum(docs[d].values())) for d in indices] for w in q if w in vocab])
    scores = np.multiply.reduce(((1.0-tlambda)*freq_match)+(tlambda*prob_words))
    ixs = np.argsort(scores)[::-1]
    ranked = [indices[i] for i in ixs ]

    try: 
        r =  ranked.index(indx)
    except ValueError:
        print "ERROR: %s" % str(q)
        r = 100
    
    return r

def find_ranks(queries,indices,docs,probtable,tlambda,length,arank):
    """iterate through the testing data and find the ranks

    :param queries: queries from the test data
    :param indices: indices of the rank list
    :param docs: the total number of signatures to rank
    :param probtable: word frequencies estimated from training data
    :param tlambda: the lambda parameter used when ranking
    :param length: number of signatures in rank list
    :param arank: the number to computer accuracy at
    """
    n = 0
    rsum = 0
    at_one = 0
    at_j = 0
    
    for (query,gold_indx) in queries:
        n += 1
        frank = rank(query.split(),gold_indx,indices,docs,tlambda,length,probtable)
        rsum += 1.0/(frank+1)
        if frank == 0:
            at_one += 1
        if frank <= 9:
            at_j += 1 
        
    mrr = rsum/n
    acc1 = float(at_one)/n
    accJ = float(at_j)/n
    return (mrr,acc1,accJ)

def write_results(out,rank,proj_dir):
    """writes the baseline result to rank_results file

    :param out: different results to report
    :type out: tuple
    :param rank: the rank to measure accuracy at
    :type rank: int
    :param proj_dir: the project directory to print results at
    :type proj_dir: str
    """
    results_file = os.path.join(proj_dir,"rank_results_baseline.txt")
    mode = 'w'
    if os.path.isfile(results_file):
        mode = 'a'

    with codecs.open(results_file,mode,encoding='utf-8') as results:
        print >>results,"term matching baseline"
        print >>results,"accuracy@1:  %f" % (out[1])
        print >>results,"accuracy@%d: %f" % (rank,out[2])
        print >>results,"MRR: %f" % out[0]

def main(config):
    tlambda = config.tlambda if config.tlambda else 0.7
    arank = config.ranksize if config.ranksize else 10
    documents,original,length = parse_rank_file(config.rfile,config.encoding)
    indices = documents.keys()
    queries = parse_queries(config.atraining,original,config.encoding)
    prob_table = find_word_occurences(config.atraining,config.encoding)
    mrr,acc1,accJ = find_ranks(queries,indices,documents,prob_table,tlambda,length,arank)
    write_results((mrr,acc1,accJ),arank,config.dir)

run = main

if __name__ == "__main__":
    main(sys.argv[1:])
    
