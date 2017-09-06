## a small testing framework for zubr lisp

import time
import os
import re
import sys
from zubr.zubr_lisp.Reader import LispReader,list_to_str 
from zubr.zubr_lisp.Lang import query_from_list,query

BAN1,BAN2,BAN3 = ["="*51,"-"*32,"+"*51]

TEST_FUNCTIONS = [
    u"test-assert-equal",
    u"test-assert",
    u"test-assert-not-equal",
    u"test-assert-true",
    u"test-assert-false",
    u"test-assert-none",
]

def __find_files(path):
    """Find if path is a testing file or a directory with test files

    :param: path to test file or test dir
    :type param: str 
    :raises: ValueError
    :rtype: list
    """
    is_file = os.path.isfile(path)
    is_dir  = os.path.isdir(path)
    if not is_file and not is_dir:
        raise ValueError('Uknown testing file or directory: %s' % path)

    if is_file: return [is_file]
    return [os.path.join(path,i) for i in os.listdir(path) if '.lisp' in i]

def __new_testfile(path,num):
    """Prints tesfile information 

    :rtype: None 
    """ 
    print >>sys.stdout, "TESTING_FILE_%d (%s)" % (num,path)
    print >>sys.stdout, BAN1

def __show_results(etime,passed,total):
    """Prints information about the test 
    
    :param etime: elapsed time
    :param passed: the total number passed
    :param total: the total number
    :rtype: None
    """
    rat = 0 if passed == 0 else float(passed)/float(total)
    print >>sys.stdout,"\nTEST RESULTS"
    print >>sys.stdout,BAN3
    print >>sys.stdout,"Tested %d items in %f seconds" % (total,etime)
    print >>sys.stdout,"Passed %d/%d tests (%f)" % (passed,total,rat)

def __test_out(ex_num,result,list_rep):
    """Prints out pass or not

    :rtype: None 
    """
    print >>sys.stdout, "Finished item %d....." % ex_num
    if result.result is True:
        print >>sys.stdout, "Test result=Pass"
    else:
        str_rep = list_to_str(list_rep)
        print >>sys.stdout, "Test result=Fail"
        print >>sys.stdout, "Assertion: %s" % str_rep

    if result.msg: 
        print >>sys.stdout, result.msg
    print >>sys.stdout,BAN2


class RunError(object):
    """A runtime error with lisp"""
    
    def __init__(self,e):
        self.result = False
        self.msg = e
    
def make_test(target):
    """Test a given test file or directory of files

    :param target: test file or directory with test file 
    :type target: str 
    :rtype: None 
    """
    ## load the test library
    query("""(load "test.lisp")""")
    test_files = __find_files(target)
    test_items = 0
    start_time = time.time()
    passed = 0
    
    for file_num,test_file in enumerate(test_files):
        if re.search(r'\.\#',test_file): continue 
        
        try: 
            contents = LispReader.parse(test_file)
        except Exception as e:
            print >>sys.stdout,"Error reading file: %s,\n %s" %\
              (test_file,e)
            continue 
            
        __new_testfile(test_file,file_num+1)
        file_tests = 1

        ## individual file 
        for file_item in contents:
            function_name = file_item[0]

            ## is not a test unfciont
            if function_name not in TEST_FUNCTIONS:
                query_from_list(file_item)

            ## actual test examples 
            else:
                try:
                    result = query_from_list(file_item)
                except Exception as e:
                    result = RunError(e)
                finally: 
                    __test_out(file_tests,result,file_item)
                    if result.result: passed += 1
                    test_items += 1
                    file_tests += 1
                
    time_elapsed = time.time()-start_time
    __show_results(time_elapsed,passed,test_items)

if __name__ == "__main__":
    pass 
