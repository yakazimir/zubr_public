## simple command-line interpret for zubr lisp

import sys
import cmd
import re
import traceback
from zubr.zubr_lisp.Lang import query as lisp_query
from zubr.zubr_lisp.Lang import parse as lisp_parse

class LispRepl(cmd.Cmd):
    _prompt = 'zubr >>> '

    def __call__(self):
        try: 
            self.prompt = LispRepl._prompt
            self.cmdloop()
        except KeyboardInterrupt:
            exit('\nbye...')

    def parseline(self,line):
        """pass to execute function if a s-expression

        :param line: the line to parse
        :type line: str
        """
        line = line.strip()
        if not line:
            return cmd.Cmd.parseline(self,line)
        first = line.split()[0]

        if first in ['EOF','quit','help','execute','parse','exit']:
            return cmd.Cmd.parseline(self,line)
        
        return ('execute',line,line)
        
    def cmdloop(self, intro='Welcome to zubr lisp\n'):
        return cmd.Cmd.cmdloop(self,intro)

    def emptyline(self):
        pass

    def do_execute(self,args):
        """executes a lisp s-expression in zubr lisp"""
        try: 
            print "=> " + str(lisp_query(args))
        except Exception,e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    def do_EOF(self, line):
        "exits the program"
        exit('\nbye...')

    def do_quit(self, args):
        """quits the program"""
        raise SystemExit

    do_exit = do_quit

    def do_parse(self,args):
        """parse an s-expression to python list datastructure

        :param args: the input arguments
        """
        try:
            print lisp_parse(args)
        except Exception,e:
            print "Error encountered: %s" % e


if __name__ == '__main__':
    l = LispRepl()()
