#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_equal,ok_,raises,assert_not_equal
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


class TestCorpusSuffixArray(object):

    def __init__(self):
        self.parallel_corpus = ParallelCorpusSuffixArray.build_array('my_corpus',EN,FR,[])
        self.english = [i.strip() for i in cinput if i]
        self.foreign = [i.strip() for i in finput if i]

    def test_initialization(self):
        ParallelCorpusSuffixArray.build_array('my_corpus',EN,FR,[])

    def test_english_positions(self):
        elist = sorted(self.english)
        for esuffix in elist:
            i = self.parallel_corpus.query_english(esuffix)
            assert_equal(elist.index(esuffix),i.start)

    def test_foreign_positions(self):
        flist = sorted(self.foreign)
        for fsuffix in flist:
            i = self.parallel_corpus.query_foreign(fsuffix)
            assert_equal(flist.index(fsuffix),i.start)

    def test_unknown(self):
        assert_equal(self.parallel_corpus.query(';').start,-1)
        assert_equal(self.parallel_corpus.query(';').end,-1)
        assert_equal(self.parallel_corpus.query('z').start,-1)
        assert_equal(self.parallel_corpus.query('z').end,-1)
        assert_equal(self.parallel_corpus.query('z',lang='fr').end,-1)
        assert_equal(self.parallel_corpus.query('z',lang='fr').start,-1)


    def test_non_ascii_query(self):
        assert_not_equal(self.parallel_corpus.query(u'фраза',lang='fr').start,-1)
        assert_not_equal(self.parallel_corpus.query_foreign('фраза').end,-1)
        assert_equal(self.parallel_corpus.query(u'ф',lang='fr').start,-1)

    ## test spans

    def test_english_span1(self):
        i = self.parallel_corpus.query('sentence')
        assert_equal((i.end-i.start)+1,3)

