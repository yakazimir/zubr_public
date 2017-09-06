from nose.tools import assert_equal,ok_,raises,assert_not_equal
from zubr.Features import Vectorizer,FeatureObj,FeatureCounter

class TestVectorizer(object):

    def __init__(self):
        self.vec   = Vectorizer({10:1.4,4:4.5})
        self.empty = Vectorizer({})

    def test_size1(self):
        assert_equal(len(self.vec.feat_counts),len(self.vec.features))

    def test_size2(self):
        assert_equal(self.vec.size,2)

    def test_empty(self):
        assert_equal(self.empty.size,0)


# class TestFeatureObj(object):

#     def __init__(self):
#         self.ex = FeatureObj(10,EMPTY_TEMPLATE)

#     def test_init_size(self):
#         assert_equal(len(self.ex.features),10)

    
    # def test_init_gold(self):
    #     assert_equal(self.ex.gold_location,-1)

    # def test_init_in_beam(self):
    #     assert_equal(self.ex.in_beam,False)

    # def test_init_correct_prediction(self):
    #     assert_equal(self.ex.correct_prediction,False)


class TestFeatureCounter(object):

    def __init__(self):
        self.counter = FeatureCounter()

    def test_add(self):
        self.counter.new = 10
        assert_equal(self.counter.new,10)
        self.counter.new += 10
        assert_equal(self.counter.new,20)

    def test_set_zero(self):
        assert_equal(self.counter.something_new,0)
