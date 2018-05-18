# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""

import sys
import re
import random
import ast
import os
import math
import codecs
import logging
from collections import defaultdict
import shutil
import copy
import subprocess
## java extractor
from zubr.doc_extractor import javalang

__all__ = [
    "load_python",
    "prepare_docstring",
    "parse_python_doc",
    "py_extract",
    "load_java",
    "java_extract",
]

util_logger = logging.getLogger('zubr.doc_extractor.util')

## doc collection

class DocCollection(object):
    """Representing a python documentation collection"""
    
    def __init__(self,fdocs=[],cdocs=[],pdescr=[],class_info=defaultdict(set),lines={},fun_class=defaultdict(set)):
        self.fdocs = fdocs
        self.cdocs = cdocs
        self.pdescr = pdescr
        self.class_info = class_info
        self.linenos = lines
        self.fun_class = fun_class
        self.edocs = []

    @property
    def logger(self):
        ## instance logger
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level)

def __check_details(config):
    """Checks general properties of the configuration, i.e. that the project exists, 
    name exists, and so on

    :param the main configuration 
    :raises: ValueError 
    """
    pname = '' if not config.name else config.name
    src_loc = os.path.join(config.proj,config.src_loc)
    
    if not os.path.isdir(config.proj):
        raise ValueError('Cannot find the target project: %s' % config.proj)

    if not os.path.isdir(src_loc):
        name_src = os.path.join(config.proj,pname)
        if not os.path.isdir(name_src):
            name_src = os.path.join(config.proj,'src')
            if not os.path.isdir(name_src):
                raise ValueError('Cannot find source directory: %s' % src_loc)
        src_loc = name_src

    ## update the src_loc
    config.src_loc = src_loc

def __extract_dir(directory,extension,ignore_test):
    """Recursively extract items from a directory

    :param directory: the input directory 
    :param extension: source file extension name
    :returns: a list of file paths 
    :rtype: list
    """
    dirs  = [directory]
    files = []
    testing = set(["tests","test"])

    while True:

        ### break if you make it through all directories
        if not dirs: break 
        new_dir = dirs.pop()
        
        for nfile in os.listdir(new_dir):
            full_path = os.path.join(new_dir,nfile)

            ### another directory?
            if os.path.isdir(full_path):
                base_name = os.path.basename(full_path)
                ## ignore testing 
                if ignore_test and base_name in testing: continue
                elif ignore_test and "Test" in base_name: continue
                dirs.append(full_path)
            else:
                ext = os.path.splitext(full_path)[1][1:].strip().lower()
                if ext != extension: continue
                files.append(full_path)

    util_logger.info('Found %d source files' % len(files))
    return files

def __copy_config(config,settings):
    """Copy over global configuration values to local config 

    :param config: the global config 
    :param settings: the local config 
    """
    settings.dir_blocks    = [i.strip() for i in config.dir_blocks.split('+')]
    settings.extract_undoc = config.extract_undoc
    settings.ignore_magic  = config.ignore_magic
    settings.ignore_test   = config.ignore_test
    settings.max_args      = config.max_args
    settings.preproc       = config.preproc
    settings.print_data    = config.print_data
    settings.out_dir       = config.dir
    settings.dir           = config.dir
    settings.run_exp       = config.run_exp
    settings.prepare_fun   = config.prepare_fun
    settings.web_addr      = config.web_addr
    settings.online_addr   = config.online_addr
    settings.class_info    = config.class_info
    settings.extract_extra = config.extract_extra
    settings.proj = config.proj

def load_python(config,settings):
    """Load the list of files for parsing a python project 

    :param config: the main configuration 
    :raises: ValueError
    """
    ## make sure everything is correct
    __check_details(config)

    ## copy configuration settings
    __copy_config(config,settings)
    
    util_logger.info('Find the target files (might take some time)...')

    ## extract the src files
    candidate_files = __extract_dir(config.src_loc,"py",config.ignore_test)

    ## finally update the source location 
    settings.src_loc = config.src_loc

    ##
    if not settings.out_dir:
        util_logger.warning('No output directory specified, will not print data!')

    return candidate_files

def __all_src_files(extension,project_dir,ignore_test=True):
    """Find all source files that match a given extension

    :param extension: the particular extension to match 
    """
    cmd = """find %s -name "*.%s" """ % (project_dir,extension)
    out = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    # Get standard out and error
    (stdout, stderr) = out.communicate()
 
    # Save found files to list
    filelist = filter(None,stdout.split('\n'))
    util_logger.info('Found %d src files...' % len(filelist))

    ## remove test files?
    if ignore_test:
        util_logger.info('Filtering out test files...')
        filtered = []

        ## iterate through 
        for src_file in filelist:
            file_name = os.path.basename(src_file).lower()
            ## this is probably too simplistic 
            if 'test' in file_name: continue
            filtered.append(src_file)
        util_logger.info('Filtering resulted in %d src files' % len(filtered))
        
        return filtered
    return filelist 

def load_java(config,settings):
    """ """

    ## make sure everything is correct
    __check_details(config)

    ## copy configuration settings
    __copy_config(config,settings)

    util_logger.info('Find the target files (might take some time)...')

    ## extract the src files
    candidate_files = __all_src_files("java",config.proj,config.ignore_test)

    ## finally update the source location 
    # settings.src_loc = config.src_loc

    # ##
    if not settings.out_dir:
        util_logger.warning('No output directory specified, will not print data!')

    util_logger.info('loaded java files...')
    return candidate_files


## from the sphinx implementation
## see https://github.com/sphinx-doc/sphinx/blob/bf3f9ef3ecc94067a2f9b17f2c863e723cf3e7af/sphinx/util/docstrings.py

def prepare_docstring(s, ignore=1):
    # type: (unicode, int) -> List[unicode]
    """Convert a docstring into lines of parseable reST.  Remove common leading
    indentation, where the indentation of a given number of lines (usually just
    one) is ignored.
    Return the docstring as a list of lines usable for inserting into a docutils
    ViewList (used as argument of nested_parse().)  An empty line is added to
    act as a separator between this docstring and following content.
    """
    lines = s.expandtabs().splitlines()
    # Find minimum indentation of any non-blank lines after ignored lines.
    margin = sys.maxsize
    for line in lines[ignore:]:
        content = len(line.lstrip())
        if content:
            indent = len(line) - content
            margin = min(margin, indent)
    # Remove indentation from ignored lines.
    for i in range(ignore):
        if i < len(lines):
            lines[i] = lines[i].lstrip()
    if margin < sys.maxsize:
        for i in range(ignore, len(lines)):
            lines[i] = lines[i][margin:]
    # Remove any leading blank lines.
    while lines and not lines[0]:
        lines.pop(0)
    # make sure there is an empty line at the end
    if lines and lines[-1]:
        lines.append('')
    return lines


## lifted from here: https://github.com/openstack/rally/blob/master/rally/common/plugin/info.py#L31-L78

class DocumentationObject(object):
    """Represent different documentation components"""
    
    def __init__(self,short_descr='',long_descr='',params=[],return_d=''):
        """Initializes a document object 

        """        
        self.short    = short_descr
        self.long     = long_descr
        self.params   = params
        self.returnd  = return_d

    def __nonzero__(self):
        return self.short != ''

    @property
    def short_len(self):
        """Returns the length of the short description

        :rtype: int
        """
        return len(self.short.split())

## STOP WORDS

stops = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])
    
## REGEXES
    
PARAM_OR_RETURNS_REGEX = re.compile(":(?:param|returns)")
RETURNS_REGEX = re.compile(":returns: (?P<doc>.*)", re.S)
PARAM_REGEX = re.compile(":param (?P<name>[\*\w]+): (?P<doc>.*?)"
    "(?:(?=:param)|(?=:return)|(?=:raises)|\Z)", re.S)

comma    = re.compile(r'([a-zA-Z0-9])(\,|\;|\:)')
paren1   = re.compile(r'\(([a-zA-Z0-9\s\-\+\.]+)\)')
paren2   = re.compile(r'\[([a-zA-Z0-9\s\-\+\.]+)\]')
punc1    = re.compile(r'\s(\,|\)|\(|\?)\s')
punc2    = re.compile(r'\s(\,|\)|\(|\?|\.)$')
punc3    = re.compile(r'(\?|\!|\.|\;|\n|\\n)$')
quote1   = re.compile(r'\'([a-zA-Z\s\-]+)\'')
quote3   = re.compile(r'\"([a-zA-Z\s\-\!]+)\"')
quote2   = re.compile(r'\`|\'|\"+')
greater  = re.compile(r'&gt')
lessthan = re.compile(r'&lt')

def reindent(string):
    return "\n".join(l.strip() for l in string.strip().split("\n"))

def __preprocess(raw_text):
    """Perform some light preprocessing on text 

    :param raw_text: the text to be preprocessed
    """
    text = raw_text.lower()
    text = re.sub(comma,r'\1 ',text)
    text = re.sub(paren1,r' \1 ',text)   
    text = re.sub(r' \( | \) ',' ',text)
    text = re.sub(punc1,r' ',text)
    text = re.sub(punc2,r' ',text)
    text = re.sub(r'\`+','',text).strip()    
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\.$','',text).strip()
    return text

def parse_python_doc(dinput,preprocess=True):
    """Parse a given python documentation string 

    :param dinput: the raw documentation input 
    :param preprocess: run a simple preprocessing pipeline on text 
    """
    short_description = long_description = returns = ""
    params = []

    if dinput: 
        docstring = "\n".join(prepare_docstring(dinput))
        lines = docstring.split("\n", 1)
        short_description = lines[0]
        
        if len(lines) > 1:
            long_description = lines[1].strip()
            params_returns_desc = None

            match = PARAM_OR_RETURNS_REGEX.search(long_description)
            if match:
                long_desc_end = match.start()
                params_returns_desc = long_description[long_desc_end:].strip()
                long_description = long_description[:long_desc_end].rstrip()
            
            if params_returns_desc:
                params = [
                    {"name": name,"doc": "\n".join(prepare_docstring(doc))}
                    for name, doc in PARAM_REGEX.findall(params_returns_desc)
                    ]

                match = RETURNS_REGEX.search(params_returns_desc)
                if match:
                    returns = reindent(match.group("doc"))

            ## check if long is just a continuation of last
            if '.' in long_description and '.' not in short_description:
                ##FIX, NEEDS A SPACE!!!
                short_description += ' '+long_description.split('.')[0]
                long_description = ' '.join(long_description.split('.')[1:])

    short_description = short_description.split('\n')[0].strip()
    short_description = re.sub(r'\s+',' ',short_description.replace('.',' ').strip())

    ## perform some light preprocessing on the text
    if preprocess:
        short_description = __preprocess(short_description)
        long_description = __preprocess(long_description)
        
    return DocumentationObject(short_description,long_description,params,returns)

###

## PY GLUE GRAMMAR

PY_G = """

## small binary grammar for glueing together phrsaes


rep -> module_class function_arg

module_class -> module class

module -> 0
module -> module module

function_arg -> function arg

class -> 1
class -> class class

function -> 2
function -> function function

arg -> 3
arg -> arg arg

"""

def __parse_py_fun(node,maxargs):
    """Parse a python function.

    :param node: the main function node 
    :returns: function name with args in a tuple
    """
    fun_name = getattr(node,'name')
    raw_doc = ast.get_docstring(node)
    args = [a.id for a in node.args.args if a.id != 'self'][:maxargs]
    return (fun_name,tuple(args))

def prep_py_name(raw_name):
    """Prepare a python name (function or module) representation 

    This basically converts underscores to spaces and camel case to spaces

    :param raw_name: the raw name of the item 
    """
    new = re.sub(r'([a-z])([A-Z])',r'\1 \2',raw_name) ## remove camel case
    new = re.sub(r'\_',r' ',new)
    new = re.sub(r'\s+',' ',new)
    new = new.lower().strip()
    return new.strip()
    
class PyDocCollection(DocCollection):
    """Representing a python documentation collection"""
    

    def compute_subclasses(self):
        """Find small categories of classes in the same part of tree and 
        create equivalence classes. 
        
        :returns: a dictionary containing symbols to tuples to identifiers 
        """
        equiv_classes = []
        cfreq = defaultdict(int)

        ## prune out frequent classes, makes classes too large 
        for _,clist in self.class_info.items():
            for c in clist:
                cfreq[c] += 1

        ## ignored items 
        ignore = set([i for i in cfreq if cfreq[i] >= 30])

        ## fund the equivalence classes 
        for cinstance,clist in self.class_info.items():
            total_set = set([i for i in clist if i not in ignore])
            total_set.add(cinstance)
            if equiv_classes == []:
                equiv_classes.append(total_set)
                continue
            for item in total_set:
                for item_set in equiv_classes:
                    if item in item_set:
                        total_set = set.union(total_set,item_set)
                        
            equiv_classes = [i for i in equiv_classes if not set.intersection(i,total_set)]
            equiv_classes.append(total_set)

        ## assign equivalence classes ids
        identifier = 0
        cclass_map = {}
        reverse_map = defaultdict(set)
        
        for eqclass in equiv_classes:
            if len(eqclass) <= 1: continue
            for item in eqclass:
                cclass_map[item] = identifier
                reverse_map[identifier].add(item)
            identifier += 1

        ## find classes of functions
        for fun_name,clist in self.fun_class.items():
            if len(clist) <= 1: continue
            cids = set(filter(None,[cclass_map.get(i,None) for i in clist]))
            for cid in cids:
                cclass_map[(fun_name,cid)] = identifier
                reverse_map[identifier] = (fun_name,cid)
                identifier += 1

        return (reverse_map,cclass_map)

    def print_data(self,config):
        """Main method for printing out the data

        :param config: the main configuration, with information about output
        :rtype: None 
        """
        ## the main function data 
        main = os.path.join(config.out_dir,'main.txt')
        splits = os.path.join(config.out_dir,'splits.txt')
        random.seed(42)
        data_len = len(self.fdocs)
        ## create a random train/test/dev dataset
        indices = range(data_len)
        random.shuffle(indices)

        if config.class_info:
            rmap,cmap = class_cats = self.compute_subclasses()
            classes = os.path.join(config.out_dir,"classes.txt")
            with codecs.open(classes,'w') as clist:
                for (identifier,citems) in rmap.items():
                    if isinstance(citems,tuple):
                        citems = [str(i) for i in citems]
                    print >>clist,"%d\t%s" % (identifier,' '.join(citems))
        else:
            cmap = {}

        ## uses a 70/30 (15/15) aplit
        m = int(math.floor(data_len*0.7))
        s = int(math.floor(data_len*0.15))
        train_indices = range(0,m)
        test_indices = range(m,m+s)
        valid_indices = range(m+s,data_len)

        ## the indices in each set 
        actual_train = [indices[i] for i in train_indices]
        actual_test  = [indices[i] for i in test_indices]
        actual_valid = [indices[i] for i in valid_indices]

        pseudo_lex = set()

        ## different testing files
        etrain = os.path.join(config.out_dir,'data_pseudo.e')
        ftrain = os.path.join(config.out_dir,'data_pseudo.f')
        etrain_bow = os.path.join(config.out_dir,'data_bow.e')
        ftrain_bow = os.path.join(config.out_dir,'data_bow.f')
        etest = os.path.join(config.out_dir,'data_test.e')
        ftest = os.path.join(config.out_dir,'data_test.f')
        evalid = os.path.join(config.out_dir,'data_val.e')
        fvalid = os.path.join(config.out_dir,'data_val.f')

        ## finalized representations by index
        rep_map  = {}
        tree_map = {}
        unique_seq = set()

        ## rank list information 
        rank_list       = []
        rank_list_trees = []

        ## print the class descriptions
        symbol_descriptions = defaultdict(set)
        extract_pairs = set()
        data_info_path = os.path.join(config.out_dir,"data_info.txt")
        data_info = open(data_info_path,'w')
        print >>data_info,"#PAIRS: %d" % len(self.fdocs)

        ##make orig_data directory
        os.mkdir(os.path.join(config.out_dir,"orig_data"))
        
        ## class descriptions 
        for (class_name,description) in self.cdocs:
            if config.prepare_fun:
                class_name = prep_py_name(class_name)
            extract_pairs.add((class_name,description))
            for symbol in class_name.split():
                for word in description.split():
                    ## remove stop words
                    if word not in stops: 
                        symbol_descriptions[symbol].add(word)

        ## parameter descriptions
        for (parameter,description) in self.pdescr:
            extract_pairs.add((parameter,description))
            for word in description.split():
                if word not in stops:
                    symbol_descriptions[symbol].add(word)

        print >>data_info,"#DESCRIPTIONS: %d" % len(extract_pairs)

        ## print the description file
        descriptions = os.path.join(config.out_dir,"descriptions.txt")
        with codecs.open(descriptions,'w',encoding='utf-8') as my_descriptions:
            for (symbol,slist) in symbol_descriptions.items():
                final = "%s\t%s" % (symbol,' '.join(slist))
                try: 
                    print >>my_descriptions,final.decode('utf-8')
                except UnicodeEncodeError:
                    print >>my_descriptions,final

        words      = 0
        vocabulary = set()
        sym_vocabulary = set()
        uri_map = {}
        class_seq = {}
        originals = {}

        ## print the main data 
        with codecs.open(main,'w',encoding='utf-8') as main_data:
            with codecs.open(splits,'w') as split_info:
                for (k,(module,cinfo,name,args,description)) in enumerate(self.fdocs):
                    original = "%s\t%s\t%s\t%s" % (module,cinfo,name,' '.join(args))
                    final = "%s\t%s\t%s\t%s\t%s" % (module,cinfo,name,' '.join(args),description)
                    try: print >>main_data,final.decode('utf-8')
                    except UnicodeEncodeError: print >>main_data,final
                    if k in actual_train: print >>split_info,"train"
                    elif k in actual_test: print >>split_info,"test"
                    elif k in actual_valid: print >>split_info,"valid"
                    cseq = False 

                    if cmap:
                        cseq = True
                        if cinfo in cmap:
                            cidentifier = cmap[cinfo]
                        else:
                            cidentifier = -1
                        if (name,cidentifier) in cmap:
                            fidentifier = cmap[(name,cidentifier)]
                        else:
                            fidentifier = -1
                        ## construct the class sequence
                    ## get the adresses (if exist)
                    address = self.linenos.get((module,cinfo,name),"www.github.com")
                    ## html representation of uri to be used for search
                    html="""<tt style='background-color:#E8E8E8;'> %s <a href='%s' target="_blank">%s</a>(%s)</tt>""" %\
                      ('.'.join([module,cinfo]),address,name,','.join(args))

                    ## build final representation
                    module_rep = ' '.join(module.split('.'))
                    class_rep = ' '.join(cinfo.split('.'))
                    if config.prepare_fun:
                        class_rep = prep_py_name(class_rep)
                        name = prep_py_name(name)

                    final = "%s %s %s %s" % (module_rep,class_rep,name,' '.join(args))
                    final = re.sub(r'\s+',' ',final).strip()
                    rep_map[k] = final.strip()
                    tree_seq = [0 for i in module_rep.split()]+[1 for i in class_rep.split()]+\
                      [2 for i in name.split()]+[3 for i in args]


                    if cseq:
                        cs = [-1 for i in module_rep.split()]+[cidentifier for i in class_rep.split()]+\
                          [fidentifier for i in name.split()]+[-1 for i in args]
                        class_seq[final] = cs
                        assert len(cs) == len(final.split()),"class seq mismatch"

                    assert len(final.split()) == len(tree_seq),"Tree mismatch"
                    tree_map[final] = tree_seq
                    uri_map[final] = (html,description.capitalize()+".")
                    originals[final] = original

                    if final not in unique_seq:
                        unique_seq.add(final)
                        rank_list.append(final)

                    ## putting training symbols into the pseudolex
                    for symbol in final.split():
                        if k in actual_train:
                            pseudo_lex.add(symbol)
                        sym_vocabulary.add(symbol)

                    ## english vocabulary
                    for word in description.split():
                        words += 1
                        vocabulary.add(word)

                #### EXTRA FUNCTIONS ADD
                if config.extract_extra:
                    for (module,cinfo,name,args,_) in self.edocs:

                        ## extract class info (if available)  
                        cseq = False 
                        if cmap:
                            cseq = True
                            if cinfo in cmap:
                                cidentifier = cmap[cinfo]
                            else:
                                cidentifier = -1
                            if (name,cidentifier) in cmap:
                                fidentifier = cmap[(name,cidentifier)]
                            else:
                                fidentifier = -1
                        ## construct the class sequence
                        ## get the adresses (if exist)
                        address = self.linenos.get((module,cinfo,name),"www.github.com")
                        html="""<tt style='background-color:#E8E8E8;'> %s <a href='%s' target="_blank">%s</a>(%s)</tt>""" %\
                          ('.'.join([module,cinfo]),address,name,','.join(args))

                        ## build final representation
                        module_rep = ' '.join(module.split('.'))
                        class_rep = ' '.join(cinfo.split('.'))
                        if config.prepare_fun:
                            class_rep = prep_py_name(class_rep)
                            name = prep_py_name(name)

                        final = "%s %s %s %s" % (module_rep,class_rep,name,' '.join(args))
                        final = re.sub(r'\s+',' ',final).strip()
                        tree_seq = [0 for i in module_rep.split()]+[1 for i in class_rep.split()]+\
                          [2 for i in name.split()]+[3 for i in args]

                        if cseq:
                            cs = [-1 for i in module_rep.split()]+[cidentifier for i in class_rep.split()]+\
                            [fidentifier for i in name.split()]+[-1 for i in args]
                            class_seq[final] = cs
                            assert len(cs) == len(final.split()),"class seq mismatch"

                        assert len(final.split()) == len(tree_seq),"Tree mismatch"
                        tree_map[final] = tree_seq
                        uri_map[final] = (html,description.capitalize()+".")

                        if final not in unique_seq:
                            unique_seq.add(final)
                            rank_list.append(final)

                        ## putting training symbols into the pseudolex
                        for symbol in final.split():
                            sym_vocabulary.add(symbol)

                        ## english vocabulary
                        for word in description.split():
                            words += 1
                            vocabulary.add(word)
                
                

        ## print data information
        print >>data_info,"#WORDS: %d" % words
        print >>data_info,"#VOCAB: %d" % len(vocabulary)
        print >>data_info,"#SYMBOLS: %d" % len(sym_vocabulary)

        ## print the standard bow train data
        train_tree = os.path.join(config.out_dir,"data.tree")
        
        with codecs.open(etrain_bow,'w',encoding='utf-8') as et:
            with codecs.open(ftrain_bow,'w',encoding='utf-8') as ft:
                with codecs.open(train_tree,'w',encoding='utf-8') as my_trees: 
                    for tindex in actual_train:
                        data_pair = self.fdocs[tindex]
                        english = data_pair[-1]
                        sem_rep = rep_map[tindex]
                        try: 
                            print >>et,english.strip().decode('utf-8')
                            print >>ft,sem_rep.strip().decode('utf-8')
                        except UnicodeEncodeError:
                            print >>et,english.strip()
                            print >>ft,sem_rep.strip()
                        print >>my_trees,"%s\t4" % " ".join([str(i) for i in tree_map[sem_rep.strip()]])

        ## copy the bag of words
        shutil.copy(etrain_bow,etrain)
        shutil.copy(ftrain_bow,ftrain)

        ## add the pseudolex entries
        with codecs.open(etrain,'a',encoding='utf-8') as et:
            with codecs.open(ftrain,'a',encoding='utf-8') as ft:
                for item in pseudo_lex:
                    for i in range(5):
                        print >>et,item.strip().decode('utf-8')
                        print >>ft,item.strip().decode('utf-8')

        ## add extract parallel data
        fin_etrain = os.path.join(config.out_dir,"data.e")
        fin_ftrain = os.path.join(config.out_dir,"data.f")
        shutil.copy(etrain,fin_etrain)
        shutil.copy(ftrain,fin_ftrain)

        ## print ``extra`` data, class/parameter descriptions
        extra_pairs = os.path.join(config.out_dir,"extra_pairs.txt")
        with codecs.open(fin_etrain,'a',encoding='utf-8') as et:
            with codecs.open(fin_ftrain,'a',encoding='utf-8') as ft:
                with codecs.open(extra_pairs,'w',encoding='utf-8') as extra: 
                    for (sem,en) in extract_pairs:
                        try: 
                            print >>et,en.strip().decode('utf-8')
                            print >>ft,sem.strip().decode('utf-8')
                        except UnicodeEncodeError:
                            print >>et,en.strip()
                            print >>ft,sem.strip()

                        try: 
                            print >>extra,("%s\t%s" % (sem,en)).decode('utf-8')
                        except UnicodeEncodeError:
                            print >>extra,("%s\t%s" % (sem,en))

        ## print the test data
        with codecs.open(etest,'w',encoding='utf-8') as etest:
            with codecs.open(ftest,'w',encoding='utf-8') as ftest:
                for tindex in actual_test:
                    data_pair = self.fdocs[tindex]
                    english = data_pair[-1]
                    sem_rep = rep_map[tindex]
                    try: 
                        print >>etest,english.strip().decode('utf-8')
                        print >>ftest,sem_rep.strip().decode('utf-8')
                    except UnicodeEncodeError:
                        print >>etest,english.strip()
                        print >>ftest,sem_rep.strip()

        ## print the validation data
        with codecs.open(evalid,'w',encoding='utf-8') as evalid:
            with codecs.open(fvalid,'w',encoding='utf-8') as fvalid:
                for tindex in actual_valid:
                    data_pair = self.fdocs[tindex]
                    english = data_pair[-1]
                    sem_rep = rep_map[tindex]
                    try: 
                        print >>evalid,english.strip().decode('utf-8')
                        print >>fvalid,sem_rep.strip().decode('utf-8')
                    except UnicodeEncodeError:
                        print >>evalid,english.strip()
                        print >>fvalid,sem_rep.strip()
        ## print the rank list
        ranks = os.path.join(config.out_dir,"rank_list.txt")
        ranks_uri = os.path.join(config.out_dir,"rank_list_uri.txt")
        rank_trees = os.path.join(config.out_dir,"rank_list.tree")
        rank_classes = os.path.join(config.out_dir,"rank_list_class.txt")
        rank_orig = os.path.join(config.out_dir,"rank_list_orig.txt")
        rank_o = codecs.open(rank_orig,'w',encoding='utf-8')

        
        with codecs.open(ranks,'w',encoding='utf-8') as my_ranks:
            with codecs.open(rank_trees,'w',encoding='utf-8') as my_trees:
                with codecs.open(ranks_uri,'w',encoding='utf-8') as uri:
                    with codecs.open(rank_classes,'w',encoding='utf-8') as cl:
                        for item in rank_list:
                            try: 
                                print >>my_ranks,item.decode('utf-8')
                                print >>rank_o,originals[item.decode('utf-8')]
                            except UnicodeEncodeError:
                                print >>my_ranks,item
                                print >>rank_o,originals[item]
                            print >>my_trees,"%s\t4" % (' '.join([str(i) for i in tree_map[item]]))
                            
                            try: 
                                print >>uri,"%s\t%s" % (uri_map[item][0].decode('utf-8'),uri_map[item][1].decode('utf-8'))
                            except UnicodeEncodeError:
                                print >>uri,"%s\t%s" % (uri_map[item][0],uri_map[item][1])

                            if class_seq:
                                print >>cl,"%s" % ' '.join([str(i) for i in class_seq[item]])
            

        ## remove rank classes if not available (will confuse the subsequence extractor)
        if not class_seq:
            os.remove(rank_classes)

        ## print the glue grammar
        grammar_out = os.path.join(config.out_dir,"grammar.txt")
        with codecs.open(grammar_out,'w',encoding='utf-8') as grammar:
            print >>grammar,PY_G.decode('utf-8')
        data_info.close()
        ## set possible config values for running translation models
        
        config.atraining = os.path.join(config.out_dir,"data")
        config.rfile = os.path.join(config.out_dir,"rank_list.txt")

        ## copy over
        orig_rank_list = os.path.join(config.out_dir,"orig_data/rank_list.txt")
        orig_rank_list_tree = os.path.join(config.out_dir,"orig_data/rank_list.tree")
        orig_descriptions = os.path.join(config.out_dir,"orig_data/descriptions.txt")
        orig_crank = os.path.join(config.out_dir,"orig_data/rank_list_class.txt")
        
        shutil.copy(descriptions,orig_descriptions)
        shutil.copy(ranks,orig_rank_list)
        shutil.copy(rank_trees,orig_rank_list_tree)
        if class_seq:
            shutil.copy(rank_classes,orig_crank)
        


#############################################
# JAVA DOCUMENTATION FUNCTIONS AND UTIL     #
#############################################

class JavaFunction(object):
    """Representation of individual java function"""
    
    def __init__(self,package,classname,name,docs,args,returnv):
        self.docs      = docs
        self.package   = package
        self.classname = classname
        self.args = args
        self.returnv = returnv
        self.name = name

    @property
    def not_empty(self):
        return self.docs.description
    
class JavaCollection(DocCollection):
    """A collection of parallel function documentation pairs"""

    def print_data(self,config):
        """Main method for printing out the data

        :param config: the main configuration, with information about output
        :rtype: None 
        """
        main = os.path.join(config.out_dir,'main.txt')
        splits = os.path.join(config.out_dir,'splits.txt')
        random.seed(42)
        data_len = len(self.fdocs)
        ## create a random train/test/dev dataset
        indices = range(data_len)
        random.shuffle(indices)

        ## uses a 70/30 (15/15) aplit
        m = int(math.floor(data_len*0.7))
        s = int(math.floor(data_len*0.15))
        train_indices = range(0,m)
        test_indices = range(m,m+s)
        valid_indices = range(m+s,data_len)

        ## the indices in each set 
        actual_train = [indices[i] for i in train_indices]
        actual_test  = [indices[i] for i in test_indices]
        actual_valid = [indices[i] for i in valid_indices]

        pseudo_lex = set()

        ## different testing files
        etrain = os.path.join(config.out_dir,'data_pseudo.e')
        ftrain = os.path.join(config.out_dir,'data_pseudo.f')
        etrain_bow = os.path.join(config.out_dir,'data_bow.e')
        ftrain_bow = os.path.join(config.out_dir,'data_bow.f')
        etest = os.path.join(config.out_dir,'data_test.e')
        ftest = os.path.join(config.out_dir,'data_test.f')
        evalid = os.path.join(config.out_dir,'data_val.e')
        fvalid = os.path.join(config.out_dir,'data_val.f')

        ## finalized representations by index
        rep_map  = {}
        tree_map = {}
        unique_seq = set()

        ## rank list information 
        rank_list       = []
        rank_list_trees = []

        ## print the class descriptions
        symbol_descriptions = defaultdict(set)
        extract_pairs = set()
        data_info_path = os.path.join(config.out_dir,"data_info.txt")
        data_info = open(data_info_path,'w')
        print >>data_info,"#PAIRS: %d" % len(self.fdocs)

        ##make orig_data directory
        os.mkdir(os.path.join(config.out_dir,"orig_data"))

        ## class descriptions
        for (libpath,oclass_name,description) in self.cdocs:
            if not description: continue 
            class_name = prep_py_name(oclass_name)
            extract_pairs.add((class_name,description,"c"))
            for symbol in class_name.split():
                for word in description.split():
                    ## remove stop words
                    if word not in stops: 
                        symbol_descriptions[symbol].add(word)

        ## parameter descriptions

        for (parameter,description) in self.pdescr:
            extract_pairs.add((parameter,description,"p"))
            for word in description.split():
                if word not in stops:
                    symbol_descriptions[symbol].add(word)

        words      = 0
        vocabulary = set()
        sym_vocabulary = set()
        uri_map = {}
        class_seq = {}
        orig_map = {}
        train_params = set()
        originals = {}
        return_descriptions = {}

        ## print the main data
        with codecs.open(main,'w',encoding='utf-8') as main_data:
            with codecs.open(splits,'w') as split_info:
                for k,doc_instance in enumerate(self.fdocs):


                    ## all parts of the representation 
                    description = doc_instance.docs.description
                    args      = doc_instance.args
                    package   = doc_instance.package
                    classname = doc_instance.classname
                    name = doc_instance.name
                    returnv   = doc_instance.returnv


                    ### orig rep 
                    final = "%s\t%s\t%s\t%s\t%s\t%s" % (package,classname,returnv,name,args,description)
                    ofinal  = final
                    ofinal = ofinal.strip()
                    try: print >>main_data,final.decode('utf-8')
                    except UnicodeEncodeError: print >>main_data,final

                    ## uri
                    address = self.linenos.get((classname,name),"www.github.com")
                    html="""<tt style='background-color:#E8E8E8;'> %s <a href='%s' target="_blank">%s</a>(%s)</tt>""" %\
                      ('.'.join([package,classname]),address,name,','.join(args.split()))


                    original_rep = final
                    ## print the split information
                    if k in actual_train: print >>split_info,"train"
                    elif k in actual_test: print >>split_info,"test"
                    elif k in actual_valid: print >>split_info,"valid"
                    ## preprocessed rep
                    class_name = prep_py_name(classname)
                    name = prep_py_name(name)
                    module_rep = ' '.join(package.split('.'))
                    ## args with types
                    args = ' '.join([' '.join(a.split("=")) for a in args.split()])
                    for arg in args.split(): train_params.add(arg)
                    
                    final = "%s %s %s %s %s" % (module_rep,class_name,returnv,name,args)
                    final = re.sub(r'\s+',' ',final).strip().lower()
                    tree_seq = [0 for i in module_rep.split()]+[1 for i in class_name.split()]+\
                          [7 for i in returnv.split()]+[2 for i in name.split()]+[3 for i in args.split()]

                    uri_map[final] = (html,description.capitalize()+".")

                    assert len(tree_seq) == len(final.split())
                    tree_map[final]     = tree_seq
                    orig_map[final]     = original_rep
                    originals[final] = ofinal

                    ## add to representation map 
                    rep_map[k] = final.strip()

                    if final not in unique_seq:
                        unique_seq.add(final)
                        rank_list.append(final)

                    ## return descriptions;
                    if doc_instance.docs.return_doc:
                        return_descriptions[final] = (description,doc_instance.docs.return_doc.strip())

                    ## putting trainig symbols in pseudolex 
                    for symbol in final.split():
                        if k in actual_train:
                            pseudo_lex.add(symbol)
                        sym_vocabulary.add(symbol)

                    ## english vocabulary
                    for word in description.split():
                        words += 1
                        vocabulary.add(word)
                    

                ## by default extracts the extra stuff
                if config.extra_extra: 
                        
                    for k2,extra_instance in enumerate(self.edocs):
                        description = extra_instance.docs.description
                        args      = extra_instance.args
                        package   = extra_instance.package
                        classname = extra_instance.classname
                        name      = extra_instance.name


                        address = self.linenos.get((classname,name),"www.github.com")
                        html="""<tt style='background-color:#E8E8E8;'> %s <a href='%s' target="_blank">%s</a>(%s)</tt>""" %\
                        ('.'.join([package,classname]),address,name,','.join(args.split()))

                        class_name = prep_py_name(classname)
                        name = prep_py_name(name)
                        module_rep = ' '.join(package.split('.'))
                        ## args with types
                        args = ' '.join([' '.join(a.split("=")) for a in args.split()])
                        final = "%s %s %s %s %s" % (module_rep,class_name,returnv,name,args)

                        ## add
                        uri_map[final] = (html,description.capitalize()+".")

                        tree_seq = [0 for i in module_rep.split()]+[1 for i in class_name.split()]+\
                          [7 for i in returnv.split()]+[2 for i in name.split()]+[3 for i in args.split()]


                        assert len(tree_seq) == len(final.split())
                        tree_map[final]     = tree_seq
                        orig_map[final]     = original_rep

                        if final not in unique_seq:
                            unique_seq.add(final)
                            rank_list.append(final)

                        ## putting training symbols into the pseudolex
                        for symbol in final.split():
                            sym_vocabulary.add(symbol)

                        
                
        ## print data information
        print >>data_info,"#WORDS: %d" % words
        print >>data_info,"#VOCAB: %d" % len(vocabulary)
        print >>data_info,"#SYMBOLS: %d" % len(sym_vocabulary)

        ## print the standard bow train data
        train_tree = os.path.join(config.out_dir,"data.tree")
        train_return_descriptions = os.path.join(config.out_dir,"train_returns.txt")
        valid_return_descriptions = os.path.join(config.out_dir,"valid_returns.txt")
        test_return_descriptions = os.path.join(config.out_dir,"test_returns.txt")

        with codecs.open(etrain_bow,'w',encoding='utf-8') as et:
            with codecs.open(ftrain_bow,'w',encoding='utf-8') as ft:
                with codecs.open(train_tree,'w',encoding='utf-8') as my_trees:
                    with codecs.open(train_return_descriptions,'w',encoding='utf-8') as returns: 
                        for tindex in actual_train:
                            data_pair = self.fdocs[tindex]
                            english = data_pair.docs.description
                            sem_rep = rep_map[tindex]

                            ## return descriptions 
                            if sem_rep in return_descriptions:
                                description,return_value = return_descriptions[sem_rep]
                                print >>returns,"%s\t%s\t%s" % (sem_rep,description,return_value)

                            try: 
                                print >>et,english.strip().decode('utf-8')
                                print >>ft,sem_rep.strip().decode('utf-8')
                            except UnicodeEncodeError:
                                print >>et,english.strip()
                                print >>ft,sem_rep.strip()
                            print >>my_trees,"%s\t4" % " ".join([str(i) for i in tree_map[sem_rep.strip()]])


        ## copy the bag of words
        shutil.copy(etrain_bow,etrain)
        shutil.copy(ftrain_bow,ftrain)

        ## add the pseudolex entries
        with codecs.open(etrain,'a',encoding='utf-8') as et:
            with codecs.open(ftrain,'a',encoding='utf-8') as ft:
                for item in pseudo_lex:
                    for i in range(5):
                        print >>et,item.strip().decode('utf-8')
                        print >>ft,item.strip().decode('utf-8')

        ## add extract parallel data
        fin_etrain = os.path.join(config.out_dir,"data.e")
        fin_ftrain = os.path.join(config.out_dir,"data.f")
        shutil.copy(etrain,fin_etrain)
        shutil.copy(ftrain,fin_ftrain)

        ## print ``extra`` data, class/parameter descriptions
        extra_pairs = os.path.join(config.out_dir,"extra_pairs.txt")
        with codecs.open(fin_etrain,'a',encoding='utf-8') as et:
            with codecs.open(fin_ftrain,'a',encoding='utf-8') as ft:
                with codecs.open(extra_pairs,'w',encoding='utf-8') as extra: 
                    for (sem,en,dtype) in extract_pairs:
                        if dtype == 'p' and sem not in train_params: continue
                        sem = sem.lower()
                        try: 
                            print >>et,en.strip().decode('utf-8')
                            print >>ft,sem.strip().decode('utf-8')
                        except UnicodeEncodeError:
                            print >>et,en.strip()
                            print >>ft,sem.strip()

                        try: 
                            print >>extra,("%s\t%s" % (sem,en)).decode('utf-8')
                        except UnicodeEncodeError:
                            print >>extra,("%s\t%s" % (sem,en))

        ## print the test data
        with codecs.open(etest,'w',encoding='utf-8') as etest:
            with codecs.open(ftest,'w',encoding='utf-8') as ftest:
                with codecs.open(test_return_descriptions,'w',encoding='utf-8') as returns: 
                    for tindex in actual_test:
                        data_pair = self.fdocs[tindex]
                        #english = data_pair[-1]
                        english = data_pair.docs.description
                        sem_rep = rep_map[tindex]

                        if sem_rep in return_descriptions:
                            description,return_value = return_descriptions[sem_rep]
                            print >>returns,"%s\t%s\t%s" % (sem_rep,description,return_value)
                            
                        try: 
                            print >>etest,english.strip().decode('utf-8')
                            print >>ftest,sem_rep.strip().decode('utf-8')
                        except UnicodeEncodeError:
                            print >>etest,english.strip()
                            print >>ftest,sem_rep.strip()


        ## print the validation data
        with codecs.open(evalid,'w',encoding='utf-8') as evalid:
            with codecs.open(fvalid,'w',encoding='utf-8') as fvalid:
                with codecs.open(valid_return_descriptions,'w',encoding='utf-8') as returns: 
                    for tindex in actual_valid:
                        data_pair = self.fdocs[tindex]
                        #english = data_pair[-1]
                        english = data_pair.docs.description
                        sem_rep = rep_map[tindex]

                        if sem_rep in return_descriptions:
                            description,return_value = return_descriptions[sem_rep]
                            print >>returns,"%s\t%s\t%s" % (sem_rep,description,return_value)
                        
                        try: 
                            print >>evalid,english.strip().decode('utf-8')
                            print >>fvalid,sem_rep.strip().decode('utf-8')
                        except UnicodeEncodeError:
                            print >>evalid,english.strip()
                            print >>fvalid,sem_rep.strip()


        ## print the rank list
        ## print the rank list
        ranks = os.path.join(config.out_dir,"rank_list.txt")
        ranks_uri = os.path.join(config.out_dir,"rank_list_uri.txt")
        rank_trees = os.path.join(config.out_dir,"rank_list.tree")
        rank_classes = os.path.join(config.out_dir,"rank_list_class.txt")
        rank_orig = os.path.join(config.out_dir,"rank_list_orig.txt")
        rank_o = codecs.open(rank_orig,'w',encoding='utf-8')
        
        with codecs.open(ranks,'w',encoding='utf-8') as my_ranks:
            with codecs.open(rank_trees,'w',encoding='utf-8') as my_trees:
                with codecs.open(ranks_uri,'w',encoding='utf-8') as uri:
                    with codecs.open(rank_classes,'w',encoding='utf-8') as cl:
                        for item in rank_list:
                            try: 
                                print >>my_ranks,item.decode('utf-8')
                                print >>rank_o,originals[item.strip().decode('utf-8')]
                            except UnicodeEncodeError:
                                print >>my_ranks,item
                                print >>rank_o,originals[item.strip()]

                            print >>my_trees,"%s\t4" % (' '.join([str(i) for i in tree_map[item]]))
                            
                            try: 
                                print >>uri,"%s\t%s" % (uri_map[item][0].decode('utf-8'),uri_map[item][1].decode('utf-8'))
                            except UnicodeEncodeError:
                                print >>uri,"%s\t%s" % (uri_map[item][0],uri_map[item][1])

                            # if class_seq:
                            #     print >>cl,"%s" % ' '.join([str(i) for i in class_seq[item]])

        rank_o.close()

def __parse_java_class(raw_string):
    """Parse a java class description 

    :param raw_string: the description to parse 
    :rtype: string
    """
    if raw_string is None or not raw_string.strip():
        return ""
    descr = re.search(r'\s+\* (.+)',raw_string,re.MULTILINE)
    if not descr: return ""
    new_descr = descr.groups()[0]
    new_descr = re.sub(r'\{\@.+ (.+)\}',r' \1 ',new_descr)
    new_descr = re.sub(r'\{\@inheritDoc\}','',new_descr)
    new_descr = re.split(r'\.($|\s+)',new_descr)[0].strip()
    new_descr = re.sub(r'\s+',' ',new_descr)
    return new_descr

def __preprocess_spec(new):
    if new.strip() == 'reinitialize': new = ''
    if 'for testing only' in new: new = ''
    if len(new.split()) <= 1: new = ''
    if len(new.split()) > 50: new = ''
    if 'not yet implemented' in new: new = ''
    new = re.sub(r'see \<a href.+',' ',new)
    new = re.sub(r'\s+',' ',new)
    return new

def __java_docs(raw_string):
    """Parses a java documentation string 

    :param raw_string: the raw string to parse 
    :type raw_string: baseline or None
    :rtype: str
    """
    if raw_string is None or not raw_string.strip():
        return javalang.javadoc.DocBlock()

    ## javalang parse object 
    parsed_doc = javalang.javadoc.parse(raw_string)

    ## some simple replacements specific to java descriptions
    new = re.sub('e\.g\.','eg',parsed_doc.description)
    new = re.sub('i\.e\.','ie',parsed_doc.description)

    #new = re.sub(r'\{\@.+ (.+)\}',r' \1 ',parsed_doc.description)
    new = re.sub(r'\{\@.+ (.+)\}',r' \1 ',new)
    new = re.sub(r'\{\@inheritDoc\}','',new)
    new = re.sub(r'\{\@link ','',new)
    new = re.split(r'\.($|\s+)',new)
    new = new[0]
    new = re.sub(r'\s+',' ',new)
    ## special cases for java
    if re.search("^<code>",new): new = ''
    elif re.search("^<pre>",new): new = ''
    new = re.sub(r'\<\/?[a-zA-Z0-9]+\>',' ',new)
    new = __preprocess(new)
    ## extra processing
    new = __preprocess_spec(new)
    parsed_doc.description = new

    ## return value
    new_return = re.sub(r'\{\@.+ (.+)\}',r' \1 ',parsed_doc.return_doc) if\
      parsed_doc.return_doc else ""
    new_return = re.sub(r'\<type\>',' ',new_return)
    new_return = re.split(r'\.($|\s+)',new_return)[0].strip()
    new_return = re.sub(r'\<\/?[a-zA-Z0-9]+\>',' ',new_return)
    new_return = re.sub(r'\s+',' ',new_return)
    parsed_doc.return_doc = __preprocess(new_return)

    ## parameter descriptions
    new_descr = []
    for (aname,arg_description) in parsed_doc.params:
        newd = re.sub(r'\{\@.+ (.+)\}',r' \1 ',arg_description)
        newd = re.split(r'\.($|\s+)',newd)[0].strip()
        newd = re.sub(r'\s+',' ',newd)
        if not newd: continue
        new_descr.append((aname,newd))
        
    parsed_doc.params = new_descr
    ## return updated

    
    return parsed_doc

def __java_function(node,package,class_name):
    """Parse a java function node 

    :param node: the function node 
    """
    name        = node.name
    parameters  = node.parameters
    description = node.documentation
    return_val  = node.return_type
    rv = return_val.name if return_val else "void"

    ## sort out the arguments 
    if parameters: args =  ' '.join(["%s=%s" % (p.type.name,p.name) for p in parameters]) 
    else: args = "void"

    ## documentation
    docs = __java_docs(description)
    return JavaFunction(package,class_name,name,docs,args,rv)

def java_extract(files,settings):
    """Extract functions and class descriptions from a list of files 

    :param files: the list of files to extract from 
    :param settings: the global extraction settings 
    """
    ## counts
    tfiles = sfiles = tfuns = dfuns = 0

    ## the documentation collection 
    collection = JavaCollection()
    fdoc = collection.fdocs
    edoc = collection.edocs
    cdoc = collection.cdocs
    pdoc = collection.pdescr
    subclasses = collection.class_info
    linenos = collection.linenos
    function_classes = collection.fun_class

    

    
    ## package/function
    fun_num = set()

    for file_path in files:
        #print file_path
        if '.#' in file_path: continue 

        with codecs.open(file_path) as src_file:
            filename = getattr(src_file,"name","<string>")
            source = src_file.read()
            tfiles += 1

            try: 
                tree = javalang.parse.parse(source)
            except Exception,e:
                util_logger.error('Parse error on %s, skipping..' % file_path)
                #util_logger.error(e,exc_info=True)
                continue

            ## API location and name
            try: 
                package =  tree.package.name
            except AttributeError:
                continue

            ## url
            if settings.online_addr:
                src_loc = file_path.replace(settings.proj,'').strip()
                page_url =  settings.online_addr+src_loc
                #print os.path.join(settings.online_addr,src_loc)
            else:
                page_url = "www.github.org"

            ## iterate components
            try: 
                for _,node in tree:

                    ## class
                    if isinstance(node,javalang.tree.ClassDeclaration):

                        class_name = node.name
                        methods = node.methods

                        ## class description
                        cdescription = __preprocess(__parse_java_class(node.documentation))
                        cdoc.append((package,class_name,cdescription))

                        for method in methods:

                            lnum,_ = method.position
                            line_loc = "%s#L%d" % (page_url,lnum)
                            if (file_path,lnum) in fun_num: pass 
                            fun_num.add((file_path,lnum))

                            ## 
                            if not isinstance(method,javalang.tree.MethodDeclaration): continue
                            fdecl = __java_function(method,package,class_name)
                            if fdecl.not_empty: fdoc.append(fdecl)
                            else: edoc.append(fdecl)

                            linenos[(class_name,fdecl.name)] = line_loc
                            
                            ## argument stuff
                            
                            for (arg,description) in fdecl.docs.params:
                                if not description.strip(): continue
                                pdescription = __preprocess(description)
                                pdoc.append((arg,pdescription))

            except RuntimeError:
                util_logger.error('encountered recursion error in %s' % package)
                        
    ## log some information
    collection.logger.info('Parsed %d source files, extracted %d (/%d) functions with documentation' %\
                               (len(files),len(fdoc),len(fdoc)+len(edoc)))
    
    ## extract datasets
    if settings.print_data:
        collection.print_data(settings) 

#########################################
# PYTHON EXTRACTION UTILS AND FUNCTIONS #
#########################################
                    

def py_extract(files,settings):
    """Extract functions from a list of files

    :param files: the list of file paths 
    :params logger: An instance logger 
    """
    ## 
    collection = PyDocCollection()
    fdoc = collection.fdocs
    edoc = collection.edocs
    cdoc = collection.cdocs
    pdoc = collection.pdescr
    subclasses = collection.class_info
    linenos = collection.linenos
    function_classes = collection.fun_class

    ## counts
    tfiles = sfiles = tfuns = dfuns = 0

    for file_path in files:
        
        with codecs.open(file_path) as src_file:
            filename = getattr(src_file,'name','<string>')
            source = src_file.read()
            tfiles += 1
            
            ## get module extensions
            online_loc = ' ' if not settings.online_addr else settings.online_addr
            wo_src = filename.replace(settings.src_loc+'/','')
            module = os.path.dirname(wo_src).replace('/','.')
            module = 'core' if not module else module
            internal_loc = os.path.dirname(wo_src)
            module_full_path = os.path.join(online_loc,internal_loc)
            full_src_path =  os.path.join(module_full_path,os.path.basename(filename))

            ## get ast tree
            try:
                tree = ast.parse(source)
            except SyntaxError,e:
                collection.logger.warning('Encountered SyntaxError in %s, skipping' % filename)
                sfiles += 1
                continue

            for _,node in ast.iter_fields(tree):
                for subnode in node:
                    name = getattr(subnode,'name',None)

                    try:

                        ## class definition 
                        if isinstance(subnode,ast.ClassDef):
                            docstring = ast.get_docstring(subnode)
                            cdoc_obj = parse_python_doc(docstring,settings.preproc)

                            if cdoc_obj: 
                                cdoc.append((name,cdoc_obj.short))

                            ## base class information
                            try: 
                                base_classes = [n.id.strip() for n in subnode.bases if hasattr(n,'id') and\
                                                    n.id not in  ['object','dict','set','list','Function','defaultdict']]
                            except Exception:
                                base_classes = []
                                collection.logger.warning('Error processing subclasses for %s in %s' %\
                                                              (name,filename))

                            ## all class functions 
                            for _,subnodes in ast.iter_fields(subnode):

                                for attribute in subnodes:
                                    if isinstance(attribute,ast.FunctionDef):

                                        ## add base classes 
                                        if base_classes:
                                            for bclass in base_classes:
                                                subclasses[name.strip()].add(bclass)

                                        ## try to extract the function
                                        tfuns += 1
                                        try: 
                                            function = __parse_py_fun(attribute,settings.max_args)
                                            raw_docstring = ast.get_docstring(attribute)
                                            doc_obj = parse_python_doc(raw_docstring,settings.preproc)

                                            ## line number information
                                            line_loc = "%s#L%d" % (full_src_path,attribute.lineno)
                                            linenos[(module,name,function[0])] = line_loc
                                            
                                            ## ignore if a docstring doesn't exist 
                                            if doc_obj and doc_obj.short_len > 2 and doc_obj.short_len < 60:
                                                dfuns += 1
                                                fdoc.append((module,name,function[0],function[1],doc_obj.short))

                                                ## add class information
                                                if base_classes:
                                                    for bclass in base_classes:
                                                        function_classes[function[0]].add(bclass)
                                                
                                                ## parameter descriptions?
                                                if doc_obj.params:
                                                    for description in doc_obj.params:
                                                        var_name = description.get('name')
                                                        var_description = description.get('doc')
                                                        var_description = var_description.split('\n')[0]
                                                        if settings.preproc:
                                                            var_description = __preprocess(var_description)
                                                        pdoc.append((var_name,var_description))

                                            ## keep track of undocumented functions
                                            else:
                                                edoc.append((module,name,function[0],function[1],""))

                                        except Exception:
                                            fname = getattr(attribute,'name')
                                            collection.logger.warning('Error parsing %s function in %s' % (fname,filename))

                        ## global function defition
                        if isinstance(subnode,ast.FunctionDef):
                            tfuns += 1
                            ## try to parse function
                            try: 
                                function = __parse_py_fun(subnode,settings.max_args)
                                raw_docstring = ast.get_docstring(subnode)
                                doc_obj = parse_python_doc(raw_docstring,settings.preproc)
                                line_loc = "%s#L%d" % (full_src_path,subnode.lineno)

                                ## Ignore if a docstring doesn't exist
                                if doc_obj and doc_obj.short_len > 2 and doc_obj.short_len < 60:
                                    dfuns += 1
                                    fdoc.append((module,'',function[0],function[1],doc_obj.short))
                                    linenos[(module,'',function[0])] =  line_loc

                                    ## parameter description?
                                    if doc_obj.params:
                                        for description in doc_obj.params:
                                            var_name = description.get('name')
                                            var_description = description.get('doc')
                                            var_description = var_description.split('\n')[0]
                                            if settings.preproc: 
                                                var_description = __preprocess(var_description)
                                            pdoc.append((var_name,var_description))

                                ## back up the undocumented ones (might want to add to )
                                else:
                                    edoc.append((module,'',function[0],function[1],''))
                                    linenos[(module,'',function[0])] =  line_loc
                                
                                            
                            except Exception,e:
                                fname = getattr(subnode,'name')
                                logging.warning('Error parsing %s function in %s' % (fname,filename))
                                
                    except IndexError,e:
                        collection.logger.warning('Encountered error at node: %s' % name)
                        continue

    ## log the number of files parsed/skipped

    collection.logger.info('Parsed %d (/%d) source files, %d (/%d) functions with documentation, %d pairs' %\
                    (tfiles-sfiles,tfiles,dfuns,tfuns,len(fdoc)))

    ## print the data
    if settings.print_data:
        collection.print_data(settings)




        
