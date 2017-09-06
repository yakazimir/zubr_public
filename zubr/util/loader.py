#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

from zubr.util._util_exceptions import ZubrUtilError
#from zubr.Rule import Rule,LexicalRule,RuleClass,LexClass
from collections import defaultdict
import logging
import sys
import os
import re
import codecs
import imp

__all__ = ["read_grammar","read_lexicon","load_grammar",
           "load_module"] 

load_log = logging.getLogger('zubr.util.loader') 

class MissingFile(ZubrUtilError):
    pass

class GrammarError(ZubrUtilError):
    pass


def load_module(module_path):
    """load a particular zubr module using format:
    zubr.module1.module12.ect.. 

    :param module_path: path of module to be loaded
    :type module_path: str
    :returns: loaded module
    :rtype: module 
    """
    mod = __import__(module_path)
    for c in module_path.split('.')[1:]:
        mod = getattr(mod,c)
    return mod

def load_script(script_path,add_path=True):
    """load a zubr (python) script using ``imp'' module

    :param script_path: path to (python) script to load
    :type script_path: str
    :param add_path: add script location to system path
    :type add_path: bool
    :returns: loaded script as module
    :rtype: module 
    """
    if 'py' not in script_path:
        script_path = "%s.py" % (script_path)
    (path, name) = os.path.split(script_path)
    (name, ext) = os.path.splitext(name)
    (extfile, filename, data) = imp.find_module(name, [path])
    ab = os.path.abspath(os.path.dirname(script_path))
    if add_path: sys.path.append(ab)     
    script = imp.load_module(name, extfile, filename, data)
    ## extra dependencies
    if 'DEPS' in vars(script):
        d = os.path.abspath(vars(script)['DEPS'])
        sys.path.append(d)
    return script


def _parse_ids(id_value):
    """return normalized identifier annotations from grammar

    two types of ids id1;id2;....

    simple string : symbol
    pair of strings: symbol11,symbol12;

    :param id_value: symbol associated with grammar rule
    :type id_value: str
    :rtype: str or tuple 
    """
    if "<" not in id_value:
        return id_value
    return tuple(id_value.split(','))

def _parse_features():
    """return normalized feature annotations from grammar

    """
    return None

def read_grammar(grammar_path,lex_path,encoding='utf-8',is_distr=False):
    """Read the  provided grammar
    
    :param grammar_path: path to desired (txt) grammar
    :type grammar_path: str
    :param lex_path: path to lexical rule file
    :type lex_path: str
    :raises: GrammarError
    """
    grammar_symbols = []
    rule_class = {}
    unary_rules = []
    non_terminals = []
    overall_rules = set()
    symbol_link = {}    
    nterminals = set()
    
    ##### ------ main grammar rules --------------------------- 
        
    with codecs.open(grammar_path, encoding=encoding) as txt_grammar:
    
        # go through each rule
        for rule in txt_grammar:
            rule = rule.strip()

            # skipped over commented lines
            if re.search(r'^[\;|\#]',rule) or not rule:
                continue

            # parse the rule 
            rule_split = rule.split('\t')
            # try to fix if not properly tab delimited
            if len(rule) >= 3 and len(rule_split) < 3:
                rule_split = re.sub(r'\s{2,}','\t',rule).split()

            # parse rule
            if len(rule_split) == 3:
                count,lhs,rhs = rule_split
                identifiers = ''
                features = ''
            elif len(rule_split) == 4:
                count,lhs,rhs,identifiers = rule_split
                features = ''
            elif len(rule_split) == 5:
                count,lhs,rhs,identifiers,features = rule_split
            else:
                if len(rule.split()) >= 3:
                    raise GrammarError('rule not tab delimited: %s' % rule)
                raise GrammarError('grammar rule not well-formed: %s' % rule)

            rhs = tuple(rhs.split())
            identifiers = identifiers.split(';')
            features = features.split(';')
            for symbol in rhs:
                nterminals.add(symbol)
                
            nterminals.add(lhs)

            #### errors ---------  
            if len(rhs) > 2:
                raise GrammarError('rule not binarized: %s' % rule)
            if len(lhs.split()) > 2:
                raise GrammarError('lhs rule too long: %s' % rule)

            ####  add symbols to symbol table ---------- 
            if lhs not in grammar_symbols:
                grammar_symbols.append(lhs)
                
            identifier = grammar_symbols.index(lhs)
            if identifier not in rule_class:
                rule_class[identifier] = RuleClass(lhs)

            #### add rule to rule set --------------
            new_rule = Rule(lhs,rhs,float(count))
            if (lhs,rhs) not in overall_rules:
                rule_class[identifier].append(new_rule)
                #rule_class[identifier].add_rule(new_rule)
                overall_rules.add((lhs,rhs))
            else:
                load_log.warning('duplicate rule, ignoring: %s' % rule) 
                continue 

            ### store symbol ids ------------
            class_index = rule_class[identifier].index(new_rule) 
            for symbol_type in _parse_ids(identifiers):
                if not symbol_type: continue
                id_tuple = (identifier,class_index) 
                if symbol_type in symbol_link:
                    symbol_link[symbol_type].append(id_tuple) 
                else:
                    symbol_link[symbol_type] = [id_tuple]

            ### store feature map --------------
            ## skip fro now 

    ##### ------ lexical grammar rules ---------------------------
    lex_map = {}
    lex_symbols = []
    lex_class = {}
    overall_lex = set()
    lex_prunable = []
    
    with codecs.open(lex_path, encoding=encoding) as lex_grammar:
    
        for rule in lex_grammar:
            rule = rule.strip()
            if re.search(r'^[\;|\#]',rule) or not rule:
                continue

            ## symbol identifiers --------------------- 
            if len(rule.split()) == 1 and '=' in rule:
                gram_sym,sem_sym = rule.split('=')
                if gram_sym not in lex_prunable:
                    lex_prunable.append(gram_sym)
                     
                if gram_sym not in nterminals:
                    load_log.warning('symbol not in grammar, ignoring: %s' % rule)
                    continue
                if sem_sym not in lex_map:
                    lex_map[sem_sym] = [gram_sym]
                continue 
                    
            ## normal rewrites --------------------------
            lex_split = rule.split('\t')

            #try to fix potential tab errors 
            if len(rule) >= 3 and len(lex_split) < 3:
                rule_split = re.sub(r'\s{2,}','\t',rule).split()

            #parse rule ------------------
            if len(lex_split) == 3:
                count,lhs,word = lex_split
                lex_features = ''
            elif len(lex_split) == 4:
                count,lhs,word,features = lex_split
            else:
                raise GrammarError('lex rule mal-formed: %s' % rule)

            count = float(count)
            #word = unicode(word,encoding) 
            ## warnings -------------------
            if lhs not in nterminals:
                load_log.warning('lex lhs not in grammar, ignoring: %s' % rule)
                continue
            if (lhs,word) in overall_lex:
                load_log.warning('duplicate rule, ignoring: %s' % rule) 
                continue

            ### add rule ------------------
            if lhs not in lex_symbols:
                lex_symbols.append(lhs)
                
            lex_id = lex_symbols.index(lhs) 
            new_lex = LexicalRule(lhs,word,prob=count)
            if lex_id not in lex_class:
                lex_class[lex_id] = []

            lex_class[lex_id].append(new_lex)


    # make the final rule lists
    main_rule_list = [] 
    main_lex_list = []
    rule_lookup = {}
    lex_lookup = {}
    ## global symbol map 
    global_map = {r:k for (k,r) in enumerate(set(nterminals))} 
    global_unary = {}
    
    ### --- normalize rule weights ----------------------

    for k,sym_name in enumerate(grammar_symbols):
        rule_list = rule_class[k]
        unary_index = []
        grammar_index = {}
        index_lookup_global = {}
        for i,rule in enumerate(rule_list):
            if not rule.is_binary:
                unary_index.append(i)
                unary_g = global_map[rule.rhs1]
                if unary_g not in global_unary:
                    global_unary[unary_g] = [(k,i)]
                else:
                    global_unary[unary_g].append((k,i))
                
            rhs_ids = tuple([global_map[r] for r in rule.rhs if r != 0])
            index_lookup_global[i] = rhs_ids
            grammar_index[rhs_ids] = i

        global_id = global_map[sym_name]
        r_class = RuleClass(sym_name,rule_list=rule_list,
                            unary_index=unary_index,
                            grammar_ids=grammar_index,
                            global_id=global_id,
                            index_global=index_lookup_global)
        
        r_class.normalize_counts()
        main_rule_list.append(r_class)
        rule_lookup[sym_name] = k
        
    for k,lex_name in enumerate(lex_symbols):
        rule_list = lex_class[k]
        global_id = global_map[lex_name]
        l_class = LexClass(lex_name,rule_list=rule_list,global_id=global_id)
        l_class.normalize_counts()
        main_lex_list.append(l_class)
        lex_lookup[lex_name] = k

    # updated lex prunable to ids
    prunable = [lex_lookup[l] for l in lex_prunable]
    for key,val in lex_map.iteritems():
        lex_map[key] = [lex_lookup[v] for v in val]

    return (main_rule_list,main_lex_list,rule_lookup,lex_lookup,
            symbol_link,lex_map,global_map,{},prunable,global_unary)

def read_lexicon(lex_path):
    """read input lexicon

    :param lex_path: path to lexicon file
    :type lex_path: str
    """
    pass 
    

def load_grammar(config):
    """find and load grammar and associated utilities (e.g. models) 

    :param config: parser config object
    :rtype: ?
    """

    pass

def load_data(config):
    """parse input data
    
    :param config: parser config object
    :yields: 
    """
    pass


#if __name__ == "__main__":
#grammar = '/Users/shostakovich/projects/zubr/examples/example_grammar.grammar'
#lex = '/Users/shostakovich/projects/zubr/examples/example_lex.lex'
#read_grammar(grammar,lex)


