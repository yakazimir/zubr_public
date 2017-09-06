# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

from nose.tools import assert_equal,raises,assert_not_equal,ok_

## test the basic feature extractor method

EN = ["this is my first sentence",
      "this is another sentence of mine",
      "yet again another thing",
      "a random string here",
      "the final sentence in thi sentence list"
      ]

F = ["rep1 1 symbol symbol3",
     "rep2 2 another symbol symbol5",
     "rep3 rep1 1 symbol3",
     "symbol1 symbol1 symbol hi",
     "symbol4 hi yo another rep3 rep1"
     ]

class TestExtractorMethod(object):

    def __init__(self):
        self.elex = {"<NONE>":0}
        self.flex = {"<NONE>":0}

        for sentence in EN:
            for w in sentence.split():
                if w not in self.elex:
                    elen = len(self.elex)
                    self.elex[w] = elen

        for rep in F:
            for w in rep.split():
                if w not in self.flex:
                    flen = len(self.flex)
                    self.flex[w] = flen

        self.vector = [1,0,1,(self.elen*self.flen),
                       1,(self.elen*self.flen)]

        self.erev = {i[1]:i[0] for i in self.elex.iteritems()}
        self.frev = {i[1]:i[0] for i in self.flex.iteritems()}

    @property
    def elen(self):
        return len(self.elex)

    @property
    def flen(self):
        return len(self.flex)

    @property
    def num_features(self):
        return sum(self.vector)

    def _assign_item(self,eval,fval):
        return (eval*self.flen)+fval

    def _assign_in_set(self,eval,fval,sid):
        return 

    def _reverse_item(self,vid):
        row = vid/self.flen
        col = vid - (row*self.flen)
        return (row,col)

    def _find_start(self,point):
        return sum(self.vector[:point])

    def _find_point_vector(self,number):
        if number < 0 or\
          number > self.num_features:
          return None
        
        oindex = -1
        while True:
            for k,item in enumerate(self.vector):
                start = self._find_start(k)
                end = start+self.vector[k]

                if number in range(start,end):
                    oindex = k
                    break
                    
            if oindex > -1:
                break

        return oindex

    def _find_overall_point(self,number):
        in_vector = self._find_point_vector(number)
        ins = number - sum(self.vector[:in_vector])
        return self._reverse_item(ins)
                
    def test_single_spot(self):
        assert_equal(self._find_start(2),1)

    def test_single_spot2(self):
        assert_equal(self._find_start(0),0)

    def test_basic_map1(self):
        """map two pairs to an index then remap it back"""
        ee = self.elex["again"]
        fe = self.flex["symbol4"]
        fval = self._assign_item(ee,fe)
        l,r = self._reverse_item(fval)

        ## assertions
        assert_equal(ee,l)
        assert_equal(fe,r)
        assert_equal("again",self.erev[l])
        assert_equal("symbol4",self.frev[r])

    def test_basic_map2(self):
        """map two pairs to an index then remap it back"""
        ee = self.elex["this"]
        fe = self.flex["rep1"]
        fval = self._assign_item(ee,fe)
        l,r = self._reverse_item(fval)

        ## assertions
        assert_equal(ee,l)
        assert_equal(fe,r)
        assert_equal("this",self.erev[l])
        assert_equal("rep1",self.frev[r])

    def test_basic_map3(self):
        """map two pairs to an index then remap it back"""
        ee = self.elex["list"]
        fe = self.flex["yo"]
        fval = self._assign_item(ee,fe)
        l,r = self._reverse_item(fval)

        ## assertions
        assert_equal(ee,l)
        assert_equal(fe,r)
        assert_equal("list",self.erev[l])
        assert_equal("yo",self.frev[r])

    def test_overall_map1(self):
        ee = self.elex["<NONE>"]
        fe = self.flex["<NONE>"]
        fval = self._assign_item(ee,fe)+self._find_start(3)
        assert_equal(fval,2)

    def test_overall_map2(self):
        ee = self.elex["this"]
        fe = self.flex["rep1"]
        fval = self._assign_item(ee,fe)+self._find_start(3)
        assert_equal(fval,17)

    def test_overall_map3(self):
        ee = self.elex["this"]
        fe = self.flex["rep1"]
        fval = self._assign_item(ee,fe)+self._find_start(3)
        l,r = self._find_overall_point(fval)

        assert_equal(l,ee)
        assert_equal(r,fe)
        assert_equal("this",self.erev[l])
        assert_equal("rep1",self.frev[r])

    def test_overall_map4(self):
        ee = self.elex["this"]
        fe = self.flex["rep1"]
        fval = self._assign_item(ee,fe)+self._find_start(5)
        l,r = self._find_overall_point(fval)

        assert_equal(l,ee)
        assert_equal(r,fe)
        assert_equal("this",self.erev[l])
        assert_equal("rep1",self.frev[r])        

    def test_overall_map5(self):
        ee = self.elex["another"]
        fe = self.flex["hi"]
        fval = self._assign_item(ee,fe)+self._find_start(5)
        l,r = self._find_overall_point(fval)

        assert_equal(l,ee)
        assert_equal(r,fe)
        assert_equal("another",self.erev[l])
        assert_equal("hi",self.frev[r])    
        
