#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson


Utilities for parsing grammars. 


Main grammar should have the following form
------------------------------------------

count    lhs    rhs1 (rhs2)    (id)    (feature)  


count/prob  : float or integer
lhs         : str
rhs1        : str
rhs2        : str
id          : str or tuple of str
feature     : str+=+int/float

For example : 

1    S    Rep
1    S    Ignore
1    Rep    Pred Arg1

### example predicates 

1    Pred   pred1@c    pred1
1    Pred   pred2@c    pred2
1    Pred   pred3@c    pred3

### example arguments 

1    Arg1   arg1@c    arg1
1    Arg1   arg2@c    arg2
1    Arg1   arg3@c    arg3


Lexical grammar should the following form
---------------------------------------

count    terminal-symbol    word    (features)

"""

from zubr.util._util_exceptions import ZubrUtilError
from zubr.RuleClass import RuleInstance,LexInstance,RuleClass,LexClass
from collections import defaultdict
import numpy as np
import logging
import sys
import re
import codecs
import os
import numpy as np


__all__ = ["read_lex","read_rules","parse_grammar"]

COMMENT = re.compile('\;+|\#+')


class GrammarError(Exception):
    pass


def _normalize_tab(line):
    """try to automatically fix tab issues, replace 2 or more spaces with tab

    :param line: line to fix/modify
    :returns: normalized version of line
    :rtype: str
    """
    return re.sub(r'\s{2,}','\t',line)
    

def read_rules(rules,rinstance,binary=True):
    """read the main grammar rules

    :param rules: the main (non-lexical) grammar rules
    :type rules: str  
    :param binary: rules must be binary
    :type binary: bool
    :raises: GrammarError
    """
    rlines = rules.split('\n')
    unique_symbols = set()
    unique_rules   = set()
    non_terminals = []
    o_non_terminals = []
    order = []
    nt_counts = defaultdict(float)
    nt_rhs = defaultdict(list)

    for k,rule in enumerate(rlines):
        rule = rule.strip()
        ## non-rule line 
        if not rule or re.search(COMMENT,rule):
            continue

        rule = _normalize_tab(rule)
        parsed_rule = rule.split('\t')

        ## check rule is well-formed
        if len(parsed_rule) <= 2:
            raise GrammarError('bad rule (rule too small, line: %d): %s' % (k,rule))
        if binary and len(parsed_rule) > 3:
            raise GrammarError('bad rule (too large, line: %d): %s' % (k,rule))

        count,lhs,rhs = parsed_rule
        lhs = lhs.strip()
        rhs = tuple(rhs.strip().split())

        ## make sure that first item is a number
        try: 
            count = float(count)
        except ValueError:
            raise GrammarError('bad count in rule (not number, line: %d): %s' % (k,rule))
        ## get rid of duplicates
        if (lhs,rhs) in unique_rules:
            raise GrammarError('duplicate rules (line: %d): %s' % (k,rule))

        unique_rules.add((lhs,rhs))
        if lhs not in unique_symbols:
            non_terminals.append(lhs)
            unique_symbols.add(lhs)

        ## add other rhs rules 
        for r in rhs:
            if r not in unique_symbols:
                o_non_terminals.append(r)
            unique_symbols.add(r)

        if lhs not in set(order):
            order.append(lhs)
        nt_counts[lhs] += count
        nt_rhs[lhs].append((lhs,rhs,count))
                
    ## construct the rule classes
    nt_identifiers = {sym:k for k,sym in enumerate(non_terminals+o_non_terminals)}
    classes = []
    
    for lhs_rule in order:
        lhs_identifier = nt_identifiers[lhs_rule]
        rules = nt_rhs[lhs_rule]
        weights = []
        rule_list = []
        denominator = nt_counts[lhs_rule]

        ### iterate rules 
        for i in range(len(rules)):
            ## uses -log probabilities 
            weights.append(-np.log(rules[i][-1]/denominator))
            rhs1 = nt_identifiers[rules[i][1][0]]
            if len(rules[i][1]) == 1:
                rhs2 = -1
            else:
                rhs2 = nt_identifiers[rules[i][1][1]]
            rule_list.append(np.array([rhs1,rhs2],dtype=np.int32))
        rule_list = np.array(rule_list,dtype=np.int32)
        weights = np.array(weights,dtype='d')
        classes.append(rinstance(lhs_identifier,rule_list,weights))

    classes = np.array(classes,dtype=np.object)
    return (nt_identifiers,classes)

def read_lex_rules(lex,identifiers):
    """read the lexical grammar file

    :param lex: lexical rule str
    :type lex: str
    :param identifiers: identifiers of non-terminal symbols
    :raises: GrammarError    
    """
    llines = lex.split('\n')
    order = []
    unique_rules = set()
    nt_counts = defaultdict(float)
    nt_rhs = defaultdict(list)
    word_map = {}

    for k,rule in enumerate(llines):
        rule = rule.strip()
        ## non-rule line 
        if not rule or re.search(COMMENT,rule):
            continue
        
        rule = _normalize_tab(rule)
        parsed_rule = rule.split('\t')
        
        ## check that rule is well-formed 
        if len(parsed_rule) <= 2:
            raise GrammarError('bad lex rule (rule too small, line: %d): %s' % (k,rule))
        if len(parsed_rule) > 3:
            raise GrammarError('bad lex rule (rule too big, line: %d): %s' % (k,rule))

        count,lhs,word = parsed_rule
        lhs = lhs.strip()
        word = word.strip()

        try:
            count = float(count)
        except ValueError:
            raise GrammarError('bad count in lex rule (line: %d): %s' % (k,rule))

        ## is the lhs a known symbol or a duplicate?
        if lhs not in identifiers:
            raise GrammarError('unknown lex symbol (line: %d): %s' % (k,lhs))
        if (lhs,word) in unique_rules:
            raise GrammarError('duplicate lex rule (line: %d): %s' % (k,rule))
        
        unique_rules.add((lhs,word))
        nt_counts[lhs] += count
        nt_rhs[lhs].append((lhs,word,count))
        if lhs not in set(order):
            order.append(lhs)
        if word not in word_map:
            word_map[word] = len(word_map)

    ## construct classes
    classes = []

    for lhs_rule in order:
        lhs_identifier = identifiers[lhs_rule]
        weights = []
        rule_list = []
        denominator = nt_counts[lhs_rule]
        rules = nt_rhs[lhs_rule]

        ## iteratate rhs rules 
        for i in range(len(rules)):
            weights.append(-np.log(rules[i][-1]/denominator))
            word = word_map[rules[i][1]]
            rule_list.append(np.array([word],dtype=np.int32))
            
        rule_list = np.array(rule_list,dtype=np.int32)
        weights = np.array(weights,dtype='d')
        classes.append(LexInstance(lhs_identifier,rule_list,weights))

    classes = np.array(classes,dtype=np.object)
    return (word_map,classes)

def find_unaries(classes):
    """find unary rules and creates a map rhs -> [(class_id,unary_id),...]"""
    unary_lookup = {}
    
    for i in range(classes.num_classes):
        rclass = classes.classes[i]
        for k,unary in enumerate(rclass._unaries):
            rhs = rclass.rules[unary][0]
            if rhs not in unary_lookup:
                unary_lookup[rhs] = [(i,unary)]
            else:
                unary_lookup[rhs].append((i,unary))

    return unary_lookup
    
def parse_grammar(config):
    """parse an input grammar (lexicon and rule file)

    - parsers assume two types of input files
        -- "name".grammar : the main (non-lexical) rules
        -- "name".lex : the lexical rules

    - config.grammar should specify /path/to/grammar/name
        
    :param config: parser configuration
    :type config: zubr.util.config.ConfigAttrs
    """
    if not config.grammar:
        raise GrammarError('please specify grammar...')
    
    name       = config.grammar
    ptype      = config.ptype.lower()
    main_rules = "%s.grammar" % name
    lex_rules  = "%s.lex" % name
    encoding   = config.encoding 

    ## binary rules?
    binary = True if 'cky' in ptype else False
    rinstance = RuleInstance(ptype)
    
    ### Do the files actuall exist?

    if not os.path.isfile(main_rules):
        raise GrammarError('cannot find rule file: %s' % main_rules)

    if not os.path.isfile(lex_rules):
        raise GrammarError('cannot file lex file: %s' % main_rules)

    rules  = codecs.open(main_rules,encoding=encoding).read()
    lrules = codecs.open(lex_rules,encoding=encoding).read()
    ## read the main rules 
    identifiers,main_classes = read_rules(rules,rinstance,binary)
    main_classes = RuleClass(main_classes)
    ## read the lex rules
    lex_identifiers,lex_classes = read_lex_rules(lrules,identifiers)
    lex_classes = LexClass(lex_classes)
    ## create a unary lookup map
    unary_map = find_unaries(main_classes)
    return (identifiers,lex_identifiers,main_classes,lex_classes,unary_map)
    

def parse_str_grammar(grammar_str):
    """parse a grammar from a string input"""
    gram_parsed = grammar_str.split("<LEX>")
    if len(gram_parsed) != 2:
        raise GrammarError('cannot parse out main/lex rules')

    ## infer type of grammar
    main_rules,lex_rules = gram_parsed
    rinstance = RuleInstance('cky')
    identifiers,main_classes = read_rules(main_rules,rinstance)
    main_classes = RuleClass(main_classes)
    lex_identifiers,lex_classes = read_lex_rules(lex_rules,identifiers)
    lex_classes = LexClass(lex_classes)
    return (identifiers,lex_identifiers,main_classes,lex_classes)


    

# def parse_feature(feature):
#     """parse a feature

#     :param feature: feature to be parsed
#     :type feature: str
#     :rtype: str
#     """
#     pass 

# def parse_identifier(identifier):
#     """parse a rule identifier"""
#     pass 

# def read_grammar(rules,lex=None,encoding='utf-8'):
#     """read the main grammar rules

#     :param rules: grammar rules from file or string
#     :type rules: str
#     :returns: grammar object
#     """
    
#     if os.path.isfile(rules):
#         rfile = codecs.open(rules,encoding=encoding)
#         rules = rfile.read()
#     else:
#         rules = unicode(rules.encode(encoding))

#     ## main rules
    
#     for rule in rules.split('\n'):
#         rule = rule.strip()
#         if re.search(r'^\#+|\;+|^\n',rule) or not rule: continue
#         rule = re.sub(r'\s{2,10}','\t',rule)
#         sections = rule.split('\t')
#         identifier = ''; features = ''

#         if len(sections) == 3:
#             count,lhs,rhs = sections

#         elif len(sections) == 4:
#             count,lhs,rhs,ids = sections

#         elif len(sections) == 5:
#             count,lhs,rhs,ids,features = sections

#         else:
#             raise GrammarError('rule malformed: %s' % rule)

#         rhs = rhs.split('\s')
        
        
    #is_file = False 
    # if os.path.isfile(rules):
    #     is_file = True
    #     rfile = codecs.open(file,encoding=encoding)
    #     rules = rfile.read()

    # for rule in rules.split('\n'):
    #     rule = rule.strip()
    #     print rule        

    # close the file 
    #if is_file:
    #    rfile.close() 
        
        
        
    





# def _construct_lexicon(lex):
#     """create a lexicon object"""
#     pass 

# def _rules_from_str(str):
#     """read grammar rules from str"""
#     pass

# def _read_rules(rule_list):
#   """read (main) grammar rules in bitpar format

#   - The format of each rule should be as follows

#   prob \t lhs \t rhs1,(rhs2) \t (sem1,sem2,...) \t (feature1,feature2,...) 

#   :param rule_list: list of grammar production rules
#   :type rule_list: list or file
#   :returns: 
#   """
#   pass 

# def _read_lex_rules(lex_list):
#   """read lexical grammar rules in bitpar format

#   :param lex_list: list of lexical productions
#   :type lex_list: list or file
#   :returns: 
#   """
#   pass 





# def loader(**kwargs):
#     """load a grammar"""

#     ## find encoding
#     if 'encoding' not in kwargs:
#         encoding = 'utf-8'
#     else:
#         encoding = kwargs['encoding'] 

#     ### load from string or file
#     if 'from_str' in kwargs:
#         pass

#     elif 'gram_path' in kwargs and 'lex_path' in kwargs:
#       grammar_file = kwargs['gram_path']
#       lex_file = kwargs['lex_path']
#       try:
#         gfile = codecs.open(grammar_file,encoding=encoding)
#         gram = gfile.readlines() 
#         lfile = codecs.open(lex_file,encoding=encoding)
#         lex = lfile.readlines() 
#       except Exception,e:
#         raise GrammarUtilError('error loading grammar: %s' % e)
      
#       gfile.close()
#       lfile.close()
      
#     else:
#         raise GrammarUtilError('Unknown grammar load options: %s' %\
#                                 str(kwargs))
