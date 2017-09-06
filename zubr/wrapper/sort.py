#!/usr/bin/env python
# -*- coding: utf-8 -*-

### wrapper for the unix sorting utilty
import os
import logging
import subprocess
import codecs
import time
import shutil
import gzip

import sys
reload(sys)
sys.setdefaultencoding('UTF8')

__all__ = [
    'sort_corpus',
    'unix_prefix_sort',
    'sort_phrase_list',
]

slogger = logging.getLogger('zubr.wrapper.sort')


def sort_phrase_list(file_path,wdir,name='phrase',nparallel=3):
    """sort a phrase list

    :param file_path: the path of the phrase file 
    :param wdir: the working directory, place to back up information 
    """
    try:
        curr_lc = os.environ['LC_ALL']
    except:
        curr_lc = ''

    ## change the locale
    slogger.info('starting the sorting, changing locale...')
    os.environ['LC_ALL'] = 'C'
    slogger.info('Local set to: %s' % os.environ['LC_ALL'])

    #slog = open("%s/phrase_sort.log" % (wdir),'w')
    slog = open("%s/%s_sort.log" % (wdir,name),'w')

    if 'linux' in sys.platform:
        parallel = '--parallel %d' % nparallel
    else:
        paralll = ''

    #out_file = os.path.join(wdir,"phrase_table_ordering.txt")
    out_file = os.path.join(wdir,"%s_table_ordering.txt" % name)
    args = "cat -n %s | sort --key=2 %s | cut -f1 > %s" % (file_path,parallel,out_file)

    slogger.info('Starting the sort subprocess, print to %s' % out_file)
    sort_time = time.time()
    p = subprocess.Popen(args,stderr=slog,shell=True)

    ## wait to complete then close 
    p.wait()
    slog.close()
    
    slogger.info('finished sorting in %s seconds' % str(time.time()-sort_time))
    ## set back LC_ALL value or remove it
    if curr_lc:
        os.environ['LC_ALL'] = curr_lc
    else:
        del os.environ['LC_ALL']
    

def unix_prefix_sort(name,prefix_file,path,out_file,by_index=True,nparallel=3):
    """Call the unix sorting utility to sort a text file, and return indices

    -- NOTE: each line index j is really j - 1 (``cat`` starts numbering lines at 1)
    
    :param path: command parameters
    :param out_file: the output file
    :param by_index: if we are only interested in the index
    :returns None
    """
    
    ## check to make sure locale is set correctly
    try:
        curr_lc = os.environ['LC_ALL']
    except:
        curr_lc = ''

    slogger.info('starting the sorting, changing locale...')
    ## should handle 'utf-8' 
    os.environ['LC_ALL'] = 'C'

    slogger.info('locale set to: %s' % os.environ['LC_ALL'])

    ## log (will report any errors encountered during sorting)
    slog = open('%s/%s_sort.log' % (path,name),'w')
        
    ## run in parallel if using linux (mac sort command doesn't have --parallel switch)
    if 'linux' in sys.platform:
        parallel = '--parallel %d' % nparallel
    else:
        parallel = ''

    #arguments
    #args = 'cat -n %s | sort --key=2 %s | cut -f1 > %s' % (prefix_file,parallel,out_file)
    args = 'gunzip -c %s | cat -n | sort --key=2 %s | cut -f1 > %s' % (prefix_file,parallel,out_file)

    ## subprocess
    slogger.info('starting the sort subprocess, printing to %s' % out_file)
    sort_time = time.time()
    p = subprocess.Popen(args,stderr=slog,shell=True)

    ## wait to complete then close 
    p.wait()
    slog.close()
    
    slogger.info('finished sorting in %s seconds' % str(time.time()-sort_time))
    ## set back LC_ALL value or remove it
    if curr_lc:
        os.environ['LC_ALL'] = curr_lc
    else:
        del os.environ['LC_ALL']

def create_suffix_file(corpus,byte_pos,size,out_file):
    """Print a file of suffixes to pass to the unix sorting algorithm

    -- Note: This is the part that takes quite a long time, must be careful
    with implementation

    :param size: the size of the corpus (measure in terms of overall words)
    :param out_file: the file to print to
    :rtype: None
    """
    slogger.info('creating temporary prefix file..')
    ftime = time.time()

    # with codecs.open(out_file,'w',encoding='utf-8') as tmp:
    #     for i in range(size):
    #         print >>tmp,corpus[byte_pos[i]:-1] # -1 because last point is a space

    with gzip.open(out_file,'wb') as tmp:
        for i in range(size):
            print >>tmp,corpus[byte_pos[i]:-1].encode('utf-8')

    slogger.info('finished building the file in %s seconds' % str(time.time()-ftime))
    # with codecs.open(out_file,'w',encoding='utf-8') as tmp:
    #     for i in range(size):
    #         print >>tmp,' '.join(corpus[i:size])

    # slogger.info('finished building the file in %s seconds' % str(time.time()-ftime))

def read_sorted_file(path):
    """Read the sorted unix file

    :param path: the location of the sorted file
    :returns: a list rendering of the file
    :rtype: list(int)
    """
    slogger.info('creating list of indices from %s' % path)
    total_indices = []

    with codecs.open(path,encoding='utf-8') as indices:
        for k,line in enumerate(indices):
            line = line.strip()
            try: 
                line_num = int(line)-1
            except ValueError:
                raise ValueError('Incorrect line input: %d\t%s' % (k,line))
            total_indices.append(line_num)

    return total_indices

def print_sorted_array(array,path,name):
    """print an already sorted array (for backup)

    :param array: the array to sort
    
    """
    sort_indices = os.path.join(path,'%s_sorted.txt' % name)
    num_suffixes = array.shape[0]

    with codecs.open(sort_indices,'w',encoding='utf-8') as indices:
        for j in range(num_suffixes):
            print >>indices,"%d" % (array[j])


def sort_corpus(name,str_corpus,byte_pos,corpus_size,work_dir,nparallel):
    """Do a prefix sort on a corpus file

    :param work_dir: the current working directory
    """
    sort_indices = os.path.join(work_dir,'%s_sorted.txt' % name)
    
    if not os.path.isfile(sort_indices):
        
        ## enumerate and print all suffixes (costly, be careful w/ implementation)
        tmp_prefixes = os.path.join(work_dir,'%s_prefixes' % name)
        #create_suffix_file(corpus,corpus_size,tmp_prefixes)
        create_suffix_file(str_corpus,byte_pos,corpus_size,tmp_prefixes)

        ## sort using a call out to the unix sort utility
        unix_prefix_sort(name,tmp_prefixes,work_dir,sort_indices,nparallel=nparallel)

        ## remove the prefix file (often very large)
        slogger.info('removing the temporary prefix file...')
        os.remove(tmp_prefixes)

    ## read the sorted file generated by unix
    total_indices = read_sorted_file(sort_indices)

    ## check that number of suffixes matches corpus size
    if len(total_indices) != corpus_size:
        slogger.error('wrong number of suffixes! %d/%d' %\
                       (len(total_indices),corpus_size))
        exit('error encountered!')
                       
    return total_indices




if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unix_prefix_sort('sort.txt','.','out.txt')
