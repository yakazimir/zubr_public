## track errors made with discriminative model
import time
import os
import sys
import codecs
import logging


def params():
    """main parameters for training a discriminative model

    :rtype: tuple
    :returns: descriptions of options with names, list of options
    """
    options = [
        ("--ex_dir","ex_dir",'',"str",
         "the experiment directory [default='']","ErrorTracker"),
        ("--name","name",'',"str",
         "(prefix) name of the data [default='']","ErrorTracker"),
     ]

    model_group = {"ErrorTracker":"track errors in validation sets"}
    return (model_group,options)

def argparser():
    """Returns an cli argument parser 

    :rtype: zubr.utill.config.ConfigObj
    :returns: default argument parser 
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr.util.track_errors [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

def __read_valid(path):
    """Read the original validation data

    :param path: the data path 
    """
    valid_data = []
    with codecs.open(path,encoding='utf-8') as my_valid:
        for line in my_valid:
            line = line.strip()
            valid_data.append(line)

    return valid_data

def __read_encoded(path):
    """Read the encoded rank data

    :param path: the path tot he encoded 
    """
    rmap = {}
    
    with codecs.open(path,encoding='utf-8') as my_ranks:
        for line in my_ranks: 
            identifier,seq,encoding = line.split('\t')
            identifier = int(identifier)
            rmap[identifier] = (seq,encoding)

    return rmap

def __find_rank_file(path):
    """Find the rank files, picks the largest number 

    :param path: path to rank result directory
    :raises: ValueError 
    """
    cand = [os.path.join(path,i) for i in os.listdir(path) if 'valid_' in i]
    last = ''
    top = 0
    for c in cand:
        num = c.split('.')[0].split('_')[-1]
        num = int(num)
        if num > top:
            last = c
    if not last:
        raise ValueError('Cannot find rank file..')

    rank_listings = []

    with codecs.open(last,encoding='utf-8') as my_ranking:
        for line in my_ranking:
            number,gold,ranks = line.split('\t')
            rank_list = [int(i) for i in ranks.split()]
            gold = int(gold)
            number = int(number)
            rank_listings.append([number,gold,rank_list])
    return rank_listings
            
    
def __check_info(config):
    """Check that information is available 

    :param config: the main configuration 
    :raises: ValueError
    """
    ## make sure that experiment directory is specified 
    if not config.ex_dir or not os.path.isdir(config.ex_dir):
        raise ValueError('Error with experiment directory')

    if not config.name:
        raise ValueError('Must specify data name..')

    ex_dir = config.ex_dir
    
    ## find the rank list
    rank_encoded = os.path.join(ex_dir,"ranks_encoded.txt")

    if not os.path.isfile(rank_encoded):
        raise ValueError('Cannot find ranks_encoded.txt file..')

    rmap = __read_encoded(rank_encoded)
    
    ## find the rank result
    rank_results = os.path.join(ex_dir,"rank_results")
    if not os.path.isdir(rank_results):
        raise ValueError('Cannot find rank_results directory...')

    rankings = __find_rank_file(rank_results)

    #rank_results =     
    
    ## find validation data
    valid_data = os.path.join(ex_dir,"orig_data/%s_val.e" % config.name)
    if not os.path.isfile(valid_data):
        raise ValueError('Cannot find the validation data')

    english = validation = __read_valid(valid_data)

    return (english,rankings,rmap)

    
def main(argv):
    """The main code execution point

    :param argv: the command line arguments: 
    :type argv: list
    """
    parser = argparser()
    config = parser.parse_args(argv[1:])
    logging.basicConfig(level=logging.DEBUG)

    ## find the validation data
    english,rankings,rmap = __check_info(config)

    ## print out the errors
    for item_num,sentence in enumerate(english):
        _,gold,rlist = rankings[item_num]
        gold_surface,gencoding = rmap[gold]
        print "="*25
        print ("Input %d: %s" % (item_num,sentence)).encode('utf-8')
        print ("Gold out: %s" % (gold_surface)).encode('utf-8')
        for k,rank in enumerate(rlist):
            front = '     '
            if rank == gold and k == 0: continue
            elif rank == gold: front = '---->'
            f = "%s\t%s\t%s" %\
              (front,rmap[rank][0],rmap[rank][1].strip())
            print f.encode('utf-8')
              

if __name__ == "__main__":
    main(sys.argv)
