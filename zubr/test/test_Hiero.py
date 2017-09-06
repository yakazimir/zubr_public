from nose.tools import assert_equal,ok_,raises,assert_not_equal
from zubr.Phrases import HieroSide,HieroRule,PhrasePair

class TestPhrasePair(object):
    def __init__(self):
        self.pair1 = PhrasePair.from_str(u'hi people\thi people*   ')

    def test_overlap1(self):
        ok_(self.pair1.word_overlap())

    # def test_ef1(self):
    #     ok_(self.pair1.sides_match())

class TestHieroSide(object):

    def __init__(self):
        self.rule1 = HieroSide(u'this is me [X] and [Y] another')
        self.empty_rule = HieroSide(u'')
        self.rule2 = HieroSide(u'this is me [X] [Y]')
        self.rule3 = HieroSide(u'this has no non-terminal')

    def test_string(self):
        assert_equal(self.rule1.string,u'this is me and another')

    def test_rules(self):
        assert_equal(self.rule1.nts,[(u'[X]',3),(u'[Y]',5)])

    def test_empty1(self):
        assert_equal(self.empty_rule.string,u'')

    def test_empty2(self):
        assert_equal(self.empty_rule.nts,[])

    def test_leftcontext(self):
        assert_equal(self.rule1.left_context,u'this is me')

    def test_rightcontext(self):
        assert_equal(self.rule1.right_context,u'another')

    def test_middlecontext(self):
        assert_equal(self.rule1.middle_context,u'and')

    def test_emptyleft(self):
        assert_equal(self.empty_rule.left_context,u'')
        
    def test_emptyright(self):
        assert_equal(self.empty_rule.right_context,u'')

    def test_emptymiddle(self):
        assert_equal(self.empty_rule.middle_context,u'')

    def test_emptymiddle2(self):
        assert_equal(self.rule2.middle_context,u'')

    def test_emptyright2(self):
        assert_equal(self.rule2.right_context,u'')

    def test_length1(self):
        assert_equal(self.rule1.context_size(),5)

    def test_length2(self):
        assert_equal(self.rule2.context_size(),3)

    def test_nont(self):
        assert_equal(self.rule3.context_size(),0)
        
class TestHieroRule(object):

    def __init__(self):
        self.rule1 = HieroRule.from_str(u"X\tthis is me [Y_2] and [X_1] another |||   [X_1] [Y_2]\t1")
        self.rule2 = HieroRule.from_str(u'X\tthis is me [X_1] ||| [X_1] this is me \t10')
        self.rule3 = HieroRule.from_str(u'X\tthis is me  |||  this is me again\t10')
        
    def test_reordering1(self):
        ok_(self.rule1.has_reordering)

    def test_reordering2(self):
        assert_not_equal(self.rule2.has_reordering,True)

    def test_reordering3(self):
        assert_not_equal(self.rule3.has_reordering,True)

    def test_contains1(self):
        assert_not_equal(self.rule1.econtainsf,True)
        assert_not_equal(self.rule1.fcontainse,True)

    def test_contains2(self):
        ok_(self.rule2.econtainsf)
        ok_(self.rule2.fcontainse)
        ok_(self.rule2.sides_match)

    def test_contains3(self):
        ok_(self.rule3.fcontainse)
        assert_not_equal(self.rule3.econtainsf,True)
        assert_not_equal(self.rule3.sides_match,True)

    def test_freq(self):
        ok_(self.rule1.freq,1)
        ok_(self.rule2.freq,10)
        ok_(self.rule3.freq,1)

    def test_tuple_rep(self):
        assert_equal(HieroRule.from_tuple(self.rule1.tuple_rep()),self.rule1)

        
        

