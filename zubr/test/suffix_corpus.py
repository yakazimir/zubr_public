#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from zubr.Datastructures import ParallelCorpusSuffixArray

EN = [[u"first", u"sentence"],[u"second", u"sentence"],[u"third", u"sentence"]]
FR = [[u"первая",u"фраза"],[u"вторая",u"фраза"],[u"третьая",u"фраза"]]

cinput=u"""
first sentence second sentence third sentence
sentence second sentence third sentence
second sentence third sentence
sentence third sentence
third sentence
sentence 
""".split('\n')


finput=u"""
первая фраза вторая фраза третьая фраза
фраза вторая фраза третьая фраза
вторая фраза третьая фраза
фраза третьая фраза
третьая фраза
фраза
""".split('\n')



if __name__ == "__main__":

    t = time.time()
    parallel_corpus = ParallelCorpusSuffixArray.build_array('en-ru',EN,FR,[])
    print time.time()-t

    cinput = filter(None,[i.strip() for i in cinput])
    finput = filter(None,[i.strip() for i in finput])

    t = time.time()
    new = sorted(cinput)
    fnew = sorted(finput)
    print time.time() - t
    
    ## puti into a proper test (test_suffix..) 
    ## check that it sorts english correctl
    # for esuffix in cinput:
    #     i = parallel_corpus.query_english(esuffix)
    #     assert new.index(esuffix) == i.start 


    # for fsuffix in finput:
    #     i = parallel_corpus.query_foreign(fsuffix)
    #     assert fnew.index(fsuffix) == i.start

    # print parallel_corpus.query('sentence').start
    # print parallel_corpus.query('sentence').end
