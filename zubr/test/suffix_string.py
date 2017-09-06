from zubr.Datastructures import SuffixArray
import time


if __name__ == "__main__":
    s = SuffixArray('amuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleyoz')
    #print s.query('z')
    #print s.query(';')
    s2 = unicode('amuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleyoz')

    t1 = time.time()

    squery = s.query
    #for i in xrange(20000):
    squery('amuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlongblablabla')

    myt = time.time()-t1
    print myt

    t2 = time.time()
    s2find = s2.find
    #for i in xrange(20000):
    s2find('amuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlonglittleamuchlongblablabla')
    
    pyt =  time.time()-t2
    print pyt

    print myt < pyt
    
    # for i in range(20000):
    #     s2.find('appleapple')
    #     #for c in s2:
    #     #   is_true = 'appleapple' in c
            
    # print time.time()-t2
