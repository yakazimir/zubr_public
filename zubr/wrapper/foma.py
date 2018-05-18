# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson


a wrapper for building finite-state networks or graphs with the help
of the FOMA toolkit 

Assumes that FOMA binaries are available at $ZUBR/bin/foma 

"""

import re
import time
import os
import logging
import subprocess
import codecs
import time
import shutil
import platform
from collections import defaultdict

wrapper_path = os.path.abspath(os.path.dirname(__file__))
util_logger = logging.getLogger('zubr.wrapper.foma')

## operating system type
TYPE = "LINUX"

if 'Darwin' in platform.platform():
    TYPE = "OSX"

class FomaWrapper(object):
    
    """A wrapper for calling foma subprocesses"""

    ## this assumes that you are calling FOMA from $ZUBR/ main dir
    FOMA = '/bin/foma/%s/foma' % TYPE

    @classmethod
    def run_script(cls,config):
        """Run a foma script given a path

        :param path: the path to the foma script
        :type path: basestring 
        """
        flog_path = os.path.join(config.dir,"foma.log")
        if os.path.isfile(flog_path): ptype = 'a'
        else: ptype = 'w'

        flog = open(flog_path,ptype)
        ## run the script
        args = ".%s -f %s" % (cls.FOMA,config.script_path)
        p = subprocess.Popen(args,stdout=flog,shell=True)
        p.wait()
        
        ## close the log file
        flog.close()

def __build_sentence_script(config,path):
    """Build a foma regex script from a sentence list

    :param config: the overall configuration
    :type config: zubr.util.config.ConfigAttrs
    :param path: path to the sentence list 
    :type path: basestring
    :rtype: None
    """
    script_path = os.path.join(config.dir,"net_build.foma")
    raw_att = os.path.join(config.dir,"raw_net.att")

    with codecs.open(script_path,'w',encoding='utf-8') as script:
        print >>script,"""echo Encoding sentence regexes\n\n"""
        num_paths = 0
        
        with codecs.open(path,encoding='utf-8') as data:
            for line in data:
                line = line.strip()
                ## LOWER CASE!
                words = ' '.join([u"%"+'%'.join(w) for w in line.lower().split()])
                regex_line = u"regex %s;" % words
                print >>script,regex_line
                num_paths += 1

        print >>script,"""echo unioning the net....\n\n"""
        print >>script,"\n\nunion net"
        print >>script,"""echo sorting the net....\n\n"""
        print >>script,"\n\nsort net"
        print >>script,"""echo writing att output\n\n"""
        print >>script,"write att > %s" % raw_att
        print >>script,"""echo number of paths %s """  % num_paths
        
    config.script_path = script_path
    config.raw_att = raw_att
    
def __expand_edges(config,new_end=True):
    """Get rid of edges with multiple labels. 
    
    :param config: the main configuration

    Explanation: 
    
    -- During the dfa minimization in foma, words 
    (standardly) are sometimes collapsed into the 
    same edge. For example, [ xor | or | and ..] x y
    will expand into something like:  

    0 1 xor
    0 1 and
    0 1 or
    1 2 x
    2 3 y

    Where the function names are labelled on the same 
    edge 0->1. 

    Our implementation requires a single word per edge, 
    so we need to expand these repeated patterns, this 
    might look something like: 

    0 1 xor 
    0 10 and
    0 11 or 
    1 2 x
    10 2 x 
    11 2 x
    2 3 y

    A similar thing is done when there are multiple final/nodes
    or states, we mapp all of these to yet another final state. 

    We can then re-load this expanded dfa and do a top sort
    to get the normal node numbers.
    """
    path = config.raw_att

    ## edges and nodes
    end_nodes = set()
    edges = set()
    nodes = set()
    lines = set()
    has_repeat = False

    ## pointers
    pointers = defaultdict(set)

    modified_path = os.path.join(config.dir,"modified_raw.att")
    modified = codecs.open(modified_path,'w',encoding='utf-8')
    base_graph = [l.strip() for l in codecs.open(path,encoding='utf-8').readlines() if l.strip()]
    config.modified_graph = modified_path

    ## find the total nodes/end nodes  
    for line in base_graph:
        if not line: continue
        if line in lines: continue
        lines.add(line)
        
        lsplit = line.split('\t')
        ## end nodes 
        if len(lsplit) == 1:
            n = int(lsplit[0]) 
            end_nodes.add(n) 
            nodes.add(n)
            continue

        start,end,word,_ = lsplit
        start = int(start); end = int(end)
        pointers[start].add((end,word))
        word = word.strip()
        nodes.add(start)
        nodes.add(end)

    ### AGAIN FOR BUILD IN NEW EDGES
    for line in base_graph:
        lsplit = line.split('\t')
        
        if len(lsplit) == 4:
            start,end,word,_ = lsplit
            start = int(start); end = int(end)
            word = word.strip()

            ## repeats 
            if (start,end) in edges:
                has_repeat = True
                new_node_id = max(nodes)+1
                nodes.add(new_node_id)

                # new edge
                print >>modified, "%d\t%d\t%s\t%s" % (start,new_node_id,word,word)
                ## print link to other stuff
                if end in end_nodes: end_nodes.add(new_node_id)
                elisty = pointers[end]
                #if new_end: 
                for (end_node,label) in elisty:
                    print >>modified, "%d\t%d\t%s\t%s" % (new_node_id,end_node,label,label)
                continue
            edges.add((start,end))
            print >>modified, line

    if new_end: 
        new_end = max(nodes)+1
        for enode in end_nodes:
            print >>modified, "%s\t%s\t*END*\t*END*" % (enode,new_end)
            #nodes.add(new_end)

    if new_end: 
        print >>modified,new_end
    else:
        print >>modified,list(end_nodes)[0]

    ## still has multiple edges? Run again
    if has_repeat:
        util_logger.info('Still has multiple edges, running again...')
        modified.close()
        m = os.path.join(config.dir,"modified_again.att")
        shutil.copy(modified_path,m)
        config.raw_att = m

        ## run again 
        __expand_edges(config,new_end=False)
        
    else:
        util_logger.info('New graph now does not have multiple edges...')

def __make_sort_script(config):
    """Make and execute a script to do a top sort on an input graph

    :param config: the main configuration 
    """
    spath = os.path.join(config.dir,"sort_script.foma")
    final = os.path.join(config.dir,"graph.att")
    paths = os.path.join(config.dir,"paths.txt")
    
    with codecs.open(spath,'w') as script:
        print >>script,"read att %s" % config.modified_graph
        print >>script,"sort net"
        print >>script,"write att > %s" % final

        ## print out the full list
        print >>script,"set print-spaced 1"
        print >>script,"print words > %s" % paths

    config.script_path = spath
    FomaWrapper.run_script(config)
    ## remove *END* from file
            
def sentence_network(config):
    """Called when building a sentence network 

    :param config: the main configuration 
    :type config: zubr.util.config.ConfigAttrs
    """
    sentence_list = os.path.join(config.dir,config.list_name)
    
    ## build a script
    if not config.raw_att: 
    
        util_logger.info('Creating the foma script...')
        __build_sentence_script(config,sentence_list)

        ## run the built script
        start = time.time()
    
        util_logger.info('Running the script and compiling the network')
        FomaWrapper.run_script(config)
        util_logger.info('Built network in %s seconds' % str(time.time()-start))

    ## get rid of edges with multiple labels?
    if config.remove_multi_edge:
        
        util_logger.info('Pruning the multiple edge labels...')
        __expand_edges(config)
        __make_sort_script(config)

    else:
        ## make the raw_att the main graph file
        pass
    
### CLI

def params():
    """Main parameters for running the foma wrapper

    :rtype: tuple
    :returns: the foma runtime parameters
    """
    options = [
         ("--task","task","sentence_net","str",
            "Build a graph from a word/sentence list [default='sentence_net']","FomaWrapper"),
         ("--remove_multi_edge","remove_multi_edge",True,"bool",
            "Removes multiple word edges [default=True]","FomaWrapper"),
        ("--list_name","list_name","rank_list.txt","str",
            "the name of the sentence list [default='rank_list.txt']","FomaWrapper"),
        ("--raw_att","raw_att",'',"str",
             "The raw att file to use (if one exists) [default='']","FomaWrapper"),
    ]

    foma_group = {
        "FomaWrapper" : "settings for the foma wrapper",
    }

    return (foma_group,options)

def main(config):
    """Main executing method

    :param config: the overall configuration
    """

    ## builds a sentence network 
    if config.task == "sentence_net":
        util_logger.info('Building a sentence dfa network...')
        sentence_network(config)
    
    else:
        raise NotImplementedError('FOMA task not implemented: %s' % config.task)

def __main__(self):
    main(sys.argv[1:])
