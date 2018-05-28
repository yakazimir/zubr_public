#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals, division

import os
import sys
import codecs
import io
import argparse
import json
import re
import shutil
from collections import defaultdict
from zubr.util.latin_encoding import small_encode

# hack for python2/3 compatibility
from io import open
argparse.open = open

class BPE(object):

    def __init__(self, codes, separator='@@', vocab=None, glossaries=None):

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.split()) for item in codes]

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []

        self.cache = {}

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        output = []
        for word in sentence.split():
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return ' '.join(output)

    def list_segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        output = []
        for word in sentence.split():
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries)]

            # print new_word
            # for item in new_word[:-1]:
            #     output.append(item + self.separator)
            # output.append(new_word[-1])
            rep = [i + self.separator for i in new_word[:-1]]+[new_word[-1]]
            output.append(rep)

        return [' '.join(seq) for seq in output]

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment, gloss)]
        return word_segments

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        #required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. The strings provided in glossaries will not be affected"+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords")

    return parser

def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries=None):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if orig in cache:
        return cache[orig]

    if orig in glossaries:
        cache[orig] = (orig,)
        return (orig,)

    if version == (0, 1):
        word = tuple(orig) + ('</w>',)
    elif version == (0, 2): # more consistent handling of word-final segments
        word = tuple(orig[:-1]) + ( orig[-1] + '</w>',)
    else:
        raise NotImplementedError

    pairs = get_pairs(word)

    if not pairs:
        return orig

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word

def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item

def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary(vocab_file, threshold):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.split()
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary

def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.

    Returns a list of subwords. In which all 'glossary' glossaries are isolated 

    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    if word == glossary or glossary not in word:
        return [word]
    else:
        splits = word.split(glossary)
        segments = [segment.strip() for split in splits[:-1] for segment in [split, glossary] if segment != '']
        return segments + [splits[-1].strip()] if splits[-1] != '' else segments

def segment_sem(config):
    bpe_dir = os.path.join(config.dir,"bpe")
    code_file = os.path.join(bpe_dir,"sem_codes.txt")
    codes = codecs.open(code_file,encoding='utf-8')
    bpe = BPE(codes,"++",None,None)
    word_info = os.path.join(bpe_dir,"sem_word_info.txt")
    new_seg = {}

    with codecs.open(word_info,'w') as info:
        orig_tokens = 0
        orig_vocab = defaultdict(int)
        new_tokens = 0
        new_vocab = defaultdict(int)

        train = config.atraining+".f"
        train_seg = config.atraining+"_seg.f"
        train_tree = config.atraining+".tree"
        tree_seq = config.atraining+"_seg.tree"
        bow_seq = config.atraining+"_bow_seg.f"

        with codecs.open(train,encoding='utf-8') as my_train:
            with codecs.open(train_seg,'w',encoding='utf-8') as new_train:
                with codecs.open(train_tree,encoding='utf-8') as trees:
                    with codecs.open(tree_seq,'w',encoding='utf-8') as new_trees:
                        #with codecs.open(bow_seq,'w',encoding='utf-8') as new_bow:
                        tree_list = [l.strip() for l in trees.readlines()]
                        for k,line in enumerate(my_train):
                            line = line.strip().lower()
                            segmented = bpe.list_segment(line.lower())
                            new_seg[line] = segmented
                            new_len = []
                            linearized = ' '.join(segmented)
                            print >>new_train,linearized

                            if k < len(tree_list):
                                #print >>new_bow,linearized
                                tree,treelen = tree_list[k].split('\t')
                                tree = [t.strip() for t in tree.split()]

                                ## find the new tree
                                assert len(segmented) == len(line.split()),"segment mismatch"
                                new_tree = []

                                for j,item in enumerate(segmented):
                                    for word in item.split():
                                        new_tree.append(tree[j])

                                ## check that new tree is correct size
                                assert len(new_tree) == len(linearized.split()),"tree mismatch"
                                tree_rep = "%s\t%s" % (' '.join(new_tree),str(treelen))
                                print >>new_trees,tree_rep
                        
                            ## original vocabulary 
                            for word in line.split():
                                word = word.strip()
                                orig_tokens += 1
                                orig_vocab[word] += 1

                            ## new vocabulary
                            for item in segmented:
                                for word in item.split(): 
                                    word = word.strip()
                                    new_tokens += 1
                                    new_vocab[word] += 1

        ## bag of words
        old_bow = config.atraining+"_bow.f"
        new_bow = config.atraining+"_bow_seg.f"

        with codecs.open(old_bow,encoding='utf-8') as my_bow:
            with codecs.open(new_bow,'w',encoding='utf-8') as new_bow:
                for line in my_bow:
                    line = line.strip()
                    ssplit = line.split(' ')
                    rest = ' '.join(ssplit[1:])
                    segmented = bpe.segment(rest).strip()
                    print >>new_bow,"%s %s" % (ssplit[0],segmented)
        
        
        ## train information
        print >>info,"#train tokens: %d" % (orig_tokens)
        print >>info,"train vocabulary: %d" % (len(orig_vocab))
        print >>info,"# train word observed once: %d" % (len([w for w,f in orig_vocab.items() if f == 1]))
        print >>info,"#bpe tokens: %d" % (new_tokens)
        print >>info,"bp vocabulary: %d" % (len(new_vocab))
        print >>info,"# bpe word observed once: %d" % (len([w for w,f in new_vocab.items() if f == 1]))

        ## valid
        ## run on validation
        valid = os.path.join(config.dir,"held_out/polyglot_val.f")
        valid_seg = os.path.join(config.dir,"held_out/polyglot_val_seg.f")
        orig_unseen = set()
        orig_tokens = 0
        new_unseen = set()
        new_tokens = 0
        
        with codecs.open(valid,encoding='utf-8') as my_test:
            with codecs.open(valid_seg,'w',encoding='utf-8') as new_test:
                for line in my_test:
                    line = line.strip().lower()
                    ## 8bit encoding, conversion out of cyrillic
                    #if config.small:
                    #    line = ' '.join([small_encode(w) for w in line.split()])

                    ssplit = line.split(' ')
                    rest = ' '.join(ssplit[1:])
                    segmented = bpe.segment(rest).strip()
                    print >>new_test,"%s %s" % (ssplit[0],segmented)

                    for word in line.split():
                        word = word.strip()
                        orig_tokens += 1
                        if word not in orig_vocab:
                            orig_unseen.add(word)
                            #orig_unseen += 1

                    for word in segmented.split():
                        word = word.strip()
                        new_tokens += 1
                        if word not in new_vocab:
                            new_unseen.add(word)
                            #new_unseen += 1

        ## print information
        print >>info,"validation tokens: %d" % orig_tokens
        print >>info,"validation bpe tokens: %d" % new_tokens
        print >>info,"# unseen test words:  %d" % (len(orig_unseen))
        print >>info,"# bpe unseen words:  %d" % (len(new_unseen))

        test = os.path.join(config.dir,"held_out/polyglot_test.f")
        test_seg = os.path.join(config.dir,"held_out/polyglot_test_seg.f")

        with codecs.open(test,encoding='utf-8') as my_test:
            with codecs.open(test_seg,'w',encoding='utf-8') as new_test:
                for line in my_test:
                    line = line.strip().lower()
                    ssplit = line.split(' ')
                    rest = ' '.join(ssplit[1:])
                    segmented = bpe.segment(rest).strip()
                    print >>new_test,"%s %s" % (ssplit[0],segmented)

        ## rank rank list
        ranks = os.path.join(config.dir,"ranks/global_rank_list.txt")
        rank_tree = os.path.join(config.dir,"ranks/rank_list.tree")
        new_ranks = os.path.join(config.dir,"ranks/global_rank_list_seg.txt")
        nrank_tree = os.path.join(config.dir,"ranks/rank_list_seg.tree")

        ## ranklist classes
        with codecs.open(ranks,encoding='utf-8') as my_ranks:
            with codecs.open(rank_tree,encoding='utf-8') as my_trees:
                with codecs.open(new_ranks,'w',encoding='utf-8') as nranks:
                    with codecs.open(nrank_tree,'w',encoding='utf-8') as ntree:
                        trees = [t.strip() for t in my_trees.readlines()]
                        for k,line in enumerate(my_ranks):
                            line = line.strip()
                            ssplit = line.split()
                            rest = ' '.join(ssplit[1:])
                            segmented = bpe.list_segment(rest.lower())
                            linearized = ' '.join(segmented)
                            print >>nranks,"%s %s" % (ssplit[0],linearized)
                            tree,treelen = trees[k].split('\t')
                            tree = [t.strip() for t in tree.split()]
                            assert len(segmented) == len(ssplit[1:]),"rankm"
                            new_tree = []
                            for j,item in enumerate(segmented):
                                for word in item.split():
                                    new_tree.append(tree[j])

                            tree_rep = "%s\t%s" % (' '.join(new_tree),str(treelen))
                            print >>ntree,tree_rep

        ### descriptions
        descriptions = os.path.join(config.dir,"descriptions_seg.txt")
        new_descriptions = os.path.join(config.dir,"descriptions_seg_seg.txt")
        if os.path.isfile(descriptions):
            with codecs.open(descriptions,encoding='utf-8') as my_descriptions:
                with codecs.open(new_descriptions,'w',encoding='utf-8') as new_descriptions:
                    new_map = defaultdict(set)
                    for line in my_descriptions:
                        line = line.strip()
                        sword,word_list = line.split('\t')
                        word_list = word_list.split()
                        segmented = bpe.segment(sword).strip()
                        for segment in segmented.split():
                            segment = segment.strip()
                            if '@@' not in segment and len(segment) <= 2: continue
                            for word in word_list:
                                word = word.strip().lower()
                                new_map[segment].add(word)

                    ##new file
                    for (sword,nlist) in new_map.items():
                        print >>new_descriptions,"%s\t%s" % (sword,' '.join(nlist))

                            
        
def segment_data(config):
    """Calling the script from zubr

    :param config: the main configuration 
    :param name: the name of the data 
    """
    bpe_dir = os.path.join(config.dir,"bpe")
    code_file = os.path.join(bpe_dir,"codes.txt")
    #vocab_file = os.path.join(bpe_dir,"vocab.txt")

    codes = codecs.open(code_file,encoding='utf-8')
    #vocab = codecs.open(vocab_file,encoding="utf-8")
    #vocab = read_vocabulary(vocab,None)
    
    bpe = BPE(codes,"@@",None,None)

    word_info = os.path.join(bpe_dir,"word_info.txt")

    with codecs.open(word_info,'w') as info: 

        orig_tokens = 0
        orig_vocab = defaultdict(int)
        new_tokens = 0
        new_vocab = defaultdict(int)

        ### run on training
        train = config.atraining+".e"
        train_seg = config.atraining+"_seg.e"
        with codecs.open(train,encoding='utf-8') as my_train:
            with codecs.open(train_seg,'w',encoding='utf-8') as new_train:
                for line in my_train:
                    
                    ## 8bit encoding, conversion out of cyrillic
                    if config.small:
                        line = ' '.join([small_encode(w) for w in line.split()])

                    segmented = bpe.segment(line.lower()).strip()
                    print >>new_train,segmented

                    ## old words 
                    for word in line.split():
                        word = word.strip() 
                        orig_tokens += 1
                        orig_vocab[word] += 1
                        
                    for word in segmented.split():
                        word = word.strip()
                        new_tokens += 1
                        new_vocab[word] += 1

        ## also segment the bag of words data 
        train_bow = config.atraining+"_bow.e"
        train_bow_seg = config.atraining+"_bow_seg.e"
        with codecs.open(train_bow,encoding='utf-8') as my_train:
            with codecs.open(train_bow_seg,'w',encoding='utf-8') as new_train:
                for line in my_train:
                    line = line.strip()
                    if config.small:
                        line = ' '.join([small_encode(w) for w in line.split()])
                    segmented = bpe.segment(line.lower()).strip()
                    print >>new_train,segmented

        ## print information
        print >>info,"#train tokens: %d" % (orig_tokens)
        print >>info,"train vocabulary: %d" % (len(orig_vocab))
        print >>info,"# train word observed once: %d" % (len([w for w,f in orig_vocab.items() if f == 1]))
        print >>info,"#bpe tokens: %d" % (new_tokens)
        print >>info,"bp vocabulary: %d" % (len(new_vocab))
        print >>info,"# bpe word observed once: %d" % (len([w for w,f in new_vocab.items() if f == 1]))
        
        ## run on test
        test = os.path.join(config.dir,"held_out/polyglot_test.e")
        test_seg = os.path.join(config.dir,"held_out/polyglot_test_seg.e")

        with codecs.open(test,encoding='utf-8') as my_test:
            with codecs.open(test_seg,'w',encoding='utf-8') as new_test:
                for line in my_test:
                    ## 8bit encoding, conversion out of cyrillic
                    if config.small:
                        line = ' '.join([small_encode(w) for w in line.split()])
                    segmented = bpe.segment(line.lower()).strip()
                    print >>new_test,segmented

        ## run on validation
        valid = os.path.join(config.dir,"held_out/polyglot_val.e")
        valid_seg = os.path.join(config.dir,"held_out/polyglot_val_seg.e")

        orig_unseen = set()
        orig_tokens = 0
        new_unseen = set()
        new_tokens = 0
        
        with codecs.open(valid,encoding='utf-8') as my_test:
            with codecs.open(valid_seg,'w',encoding='utf-8') as new_test:
                for line in my_test:
                    ## 8bit encoding, conversion out of cyrillic
                    if config.small:
                        line = ' '.join([small_encode(w) for w in line.split()])
                        
                    segmented = bpe.segment(line.lower()).strip()
                    print >>new_test,segmented

                    for word in line.split():
                        word = word.strip()
                        orig_tokens += 1
                        if word not in orig_vocab:
                            orig_unseen.add(word)
                            #orig_unseen += 1

                    for word in segmented.split():
                        word = word.strip()
                        new_tokens += 1
                        if word not in new_vocab:
                            new_unseen.add(word)
                            #new_unseen += 1

        ## print information
        print >>info,"validation tokens: %d" % orig_tokens
        print >>info,"validation bpe tokens: %d" % new_tokens
        print >>info,"# unseen test words:  %d" % (len(orig_unseen))
        print >>info,"# bpe unseen words:  %d" % (len(new_unseen))

        ### description file
        descriptions = os.path.join(config.dir,"descriptions.txt")
        ndescriptions = os.path.join(config.dir,"descriptions_seg.txt")
        if os.path.isfile(descriptions):
            with codecs.open(descriptions,encoding='utf-8') as my_descriptions:
                with codecs.open(ndescriptions,'w',encoding='utf-8') as new_descriptions:
                    new_map = defaultdict(set)
                    for line in my_descriptions:
                        line = line.strip()
                        sword,word_list = line.split('\t')
                        word_list = word_list.split()
                        for word in word_list:
                            word = word.strip().lower()
                            if config.small:
                                word = small_encode(word)
                            segmented = bpe.segment(word).strip()
                            for new_word in segmented.split():
                                if '@@' not in new_word and len(new_word) <= 2: continue
                                new_map[sword].add(new_word.strip())

                    for (sword,nlist) in new_map.items():
                        print >>new_descriptions,"%s\t%s" % (sword,' '.join(nlist))


def from_data(config):
    """Learn and apply bpe algorithm to some data for pipeline 

    :param config: the global pipeline script for a zubr run 
    """
    from zubr.util.learn_bpe import from_data as learn_bpe

    ## learn the subword candidates 
    learn_bpe(config)

    ## apply them

    ## application parser
    parser = create_parser()
    args = parser.parse_args([])

    #args.input = codecs.open(etrain,encoding='utf-8')
    #args.output = codecs.open(enew,'w',encoding='utf-8')
    info = codecs.open(os.path.join(config.dir,"BPE_INFO.txt"),'w')

    if config.make_subword:
            ## english side
        args.codes = codecs.open(os.path.join(config.dir,"codes.txt"),encoding='utf-8')
        etrain = config.atraining+"."+config.target
        enew = config.atraining+"_bpe."+config.target
        etest = config.atraining+"_test."+config.target
        enewt = config.atraining+"_test_bpe."+config.target
        e_val = config.atraining+"_val."+config.target
        enewv = config.atraining+"_val_bpe."+config.target
        bpe = BPE(args.codes,u"@@",None,None)

        ## count tokens
        val_words   = set()
        train_words = set()
        new_val_words   = set()
        new_train_words = set()
        not_in_1 = set()
        not_in_2 = set()

        with codecs.open(etrain,encoding='utf-8') as main_train:
            with codecs.open(enew,'w',encoding='utf-8') as new_train:
                for line in main_train:
                    segmented = bpe.segment(line).strip()
                    print >>new_train,segmented #bpe.segment(line).strip()
                    ## count words 
                    for word in segmented.split():
                        new_train_words.add(word.strip())
                    for word in line.split():
                        train_words.add(word.strip())

        with codecs.open(etest,encoding='utf-8') as main_test:
            with codecs.open(enewt,'w',encoding='utf-8') as new_test:
                for line in main_test:
                    print >>new_test,bpe.segment(line).strip()

        with codecs.open(e_val,encoding='utf-8') as main_val:
            with codecs.open(enewv,'w',encoding='utf-8') as new_val:
                for line in main_val:
                    segmented = bpe.segment(line).strip()
                    print >>new_val,segmented
                    ## count words 
                    for word in segmented.split():
                        new_val_words.add(word.strip())
                        if word not in train_words: not_in_1.add(word)
                    for word in line.split():
                        val_words.add(word.strip())
                        if word not in train_words:
                            not_in_2.add(word)

        ## update
        shutil.copy(enew,etrain)
        shutil.copy(enewt,etest)
        shutil.copy(enewv,e_val)

        ## print the information
        print >>info,"# e-side train words: %d" % len(train_words)
        print >>info,"# e-side train segmented words: %d" % len(new_train_words)
        print >>info,"# e-side val words: %d" % len(val_words)
        print >>info,"# e-side val segmented words: %d" % len(new_val_words)
        print >>info,"# val OOV: %d" % len(not_in_2)
        print >>info,"# val segmented OOV: %d\n\n" % len(not_in_1)

    if config.make_sem_subword:
        ## count tokens
        val_words   = set()
        train_words = set()
        new_val_words   = set()
        new_train_words = set()
        not_in_1 = set()
        not_in_2 = set()
        
        ## foreign side
        args.codecs = codecs.open(os.path.join(config.dir,"sem_codes.txt"),encoding='utf-8')
        ftrain = config.atraining+"."+config.source
        fnew = config.atraining+"_bpe."+config.source
        ftest = config.atraining+"_test."+config.source
        fnewt = config.atraining+"_test_bpe."+config.source
        fval = config.atraining+"_val."+config.source
        fnewv = config.atraining+"_val_bpe."+config.source
        rank_list = os.path.join(config.dir,"rank_list.txt")
        new_rank_list = os.path.join(config.dir,"rank_list_bpe.txt")
    
        bpe_sem = BPE(args.codecs,u"@@",None,None)
        with codecs.open(ftrain,encoding='utf-8') as main_train:
            with codecs.open(fnew,'w',encoding='utf-8') as new_train:
                for line in main_train:
                    segmented = bpe_sem.segment(line).strip()
                    print >>new_train,segmented #bpe_sem.segment(line).strip()
                    
                    ## count words 
                    for word in segmented.split():
                        new_train_words.add(word.strip())
                    for word in line.split():
                        train_words.add(word.strip())

        with codecs.open(ftest,encoding='utf-8') as main_test:
            with codecs.open(fnewt,'w',encoding='utf-8') as new_test:
                for line in main_test:
                    print >>new_test,bpe_sem.segment(line).strip()

        with codecs.open(fval,encoding='utf-8') as main_val:
            with codecs.open(fnewv,'w',encoding='utf-8') as new_val:
                for line in main_val:
                    segmented = bpe_sem.segment(line).strip()
                    print >>new_val,segmented #bpe_sem.segment(line).strip()
                    ## count words 
                    for word in segmented.split():
                        new_val_words.add(word.strip())
                        if word not in train_words:
                            not_in_1.add(word)
                            #not_in_1 += 1
                    for word in line.split():
                        val_words.add(word.strip())
                        if word not in train_words:
                            not_in_2.add(word)
                            #not_in_2 += 1

        with codecs.open(rank_list,encoding='utf-8') as main_ranks:
            with codecs.open(new_rank_list,'w',encoding='utf-8') as new_ranks:
                for line in main_ranks:
                    print >>new_ranks,bpe_sem.segment(line).strip()

        ## copy over
        shutil.copy(fnew,ftrain)
        shutil.copy(fnewt,ftest)
        shutil.copy(fnewv,fval)
        shutil.copy(new_rank_list,rank_list)

        ## print the information
        print >>info,"# f-side train words: %d" % len(train_words)
        print >>info,"# f-side train segmented words: %d" % len(new_train_words)
        print >>info,"# f-side val words: %d" % len(val_words)
        print >>info,"# f-side val segmented words: %d" % len(new_val_words)
        print >>info,"# val OOV: %d" % len(not_in_1)
        print >>info,"# val segmented OOV: %d" % len(not_in_2)


if __name__ == '__main__':

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    args.codes = codecs.open(args.codes.name, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.vocabulary:
        args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

    if args.vocabulary:
        vocabulary = read_vocabulary(args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    bpe = BPE(args.codes, args.separator, vocabulary, args.glossaries)

    for line in args.input:
        args.output.write(bpe.segment(line).strip())
        args.output.write('\n')
