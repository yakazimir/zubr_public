import codecs
import sys
import os
import re
from zubr.Phrases import HieroRule,PhrasePair


def read_hiero_grammar(path,glue_path):
    """Reads a hiero grammar and results a list of rules

    :param path: path to the hiero grammar rules 
    :param glue_path: path to the glue grammar
    """

    if not os.path.isfile(path):
        raise ValueError('cannot find the target hiero grammar: %s' % path)

    
    if not os.path.isfile(glue_path):
        raise ValueError('cannot find the glue grammar: %s' % glue_path)

    hier_rules    = {}
    grammar_table = {}
    lhs = set()
    heside = set()
    hfside = set()

    with codecs.open(path,encoding='utf-8') as hiero:
        for k,line in enumerate(hiero):
            line = line.strip()
            try:
                rule = HieroRule.from_str(line,rule_num=k)
                #hier_rules[rule.tuple_rep()] = (rule,k)
                hier_rules[rule.tuple_rep()] = (k,rule.freq)
            except Exception as e:
                raise ValueError('Error parsing hiero rule (possibly malformed): %s' % line)
            heside.add(rule.erhs.string.strip())
            hfside.add(rule.frhs.string.strip())

    with codecs.open(glue_path,encoding='utf-8') as myg:
        for line in myg:
            line = line.strip()
            if re.search(r'^\#',line) or not line: continue
            left,right = line.split(' -> ')
            rhs = tuple([i.strip() for i in right.split()])
            grammar_table[rhs] = left.strip()
            lhs.add(left.strip())

    nlhs = {i:k for k,i in enumerate(lhs)}
    heside = {i:k for k,i in enumerate(heside)}
    hfside = {i:k for k,i in enumerate(hfside)}
    return (hier_rules,grammar_table,nlhs,heside,hfside)

def read_phrase_table(path):
    """Read a prhase table and convert it to phrase objects 

    :param path: path to working directory 
    :type path: str
    :returns: a map of phrase tuple representation to phrase objects
    :rtype: tuple
    :returns: tuple of english phrases and english foreign phrase pairs
    :raises: ValueError
    """
    table_path = os.path.join(path,"phrase_table.txt")

    if not os.path.isfile(table_path):
        raise ValueError('Cannot find the target phrase table: %s' % table_path)

    phrase_rule = {}
    english_phrases = {}
    foreign_phrases = {}
    
    with codecs.open(table_path,encoding='utf-8') as phrase_table:
        for k,line in enumerate(phrase_table):
            line = line.strip()
            try: 
                rule = PhrasePair.from_str(line)
                #phrase_rule[rule.tuple_rep()] = (rule,k)
                ## just need the id, will generate the object anyways
                phrase_rule[rule.tuple_rep()] = k
                english_phrases[rule.tuple_rep()[0]] = 1
                foreign_phrases[rule.tuple_rep()[1]] = 1
            except Exception as e:
                raise ValueError('Error parsing phrase table entry: %s' % line)

    kenglish_phrases = {i.strip():k for k,i in enumerate(english_phrases)}
    kforeign_phrases = {i.strip():k for k,i in enumerate(foreign_phrases)}    
    return (phrase_rule,kenglish_phrases,kforeign_phrases)
    #return phrase_rule
