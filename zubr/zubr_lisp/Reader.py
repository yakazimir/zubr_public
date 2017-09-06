#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

reader for lisp input

A variation of Peter Norvig's lispy input port
"""

import re
import os
import codecs
import numpy as np
from StringIO import StringIO as ibuffer


ENCODING = 'utf-8'
EOF = '#<eof-object>'
PAT = r"""\s*(,@|[('`,)]|"(?:[\\].|[^\\"])*"|;.*|[^\s('"`,;)]*)(.*)"""
LFILE = r'\.[a-zA-Z0-9\_]+$'

class LispSymbol(unicode):
    pass

_quote           = LispSymbol(u'quote')
_quasiquote      = LispSymbol(u'quasiquote')
_unquote         = LispSymbol(u'unquote')
_unquotesplicing = LispSymbol(u'unquotesplicing')

quotes = {"'":_quote, "`":_quasiquote, ",":_unquote, ",@":_unquotesplicing}

## the atomic datatypes (from norvig with a few additions)

def list_to_str(list_rep):
    """Converts a list representaiton to s-expression 

    :param list_rep: a lisp list 
    :rtype: str
    """
    str_list = str(list_rep)
    
    new  = re.sub(r'\[','(',str_list)
    new = re.sub(r'\]',')',new)
    new = re.sub(r'u\'(.+)\'',r'\1',new)
    new = re.sub(r'"(.+)"',r'\1',new)
    new = re.sub(r'\,','',new)
    return new

def make_atomic(token):
    """for a given symbol, make atomic python value

    :param token: the input to convert into atomic python value
    :type token: unicode
    """
    ## booleans
    if token == u"True":
        return True
    elif token == u"False":
        return False

    elif token == u"None":
        return None

    ## infinite and nan
    elif token == u"inf":
        return np.inf
    elif token == u"nan":
        return np.nan

    ## string literal
    elif token and token[0] == '"':
        return unicode(token[1:-1])

    ## try numbers 
    try:
        return int(token)
    except ValueError:
        try: 
            return float(token)
        except ValueError:
            try:
                return complex(token.replace('i', 'j', 1))
            except ValueError:
                ## return symbol
                return LispSymbol(token)

class LispReader(object):

    """lisp reader that keep track of line point in file or stream"""

    def __init__(self,file_or_stream):
        self.file = file_or_stream
        self.line = u''
        self._lnum = 0

    @property
    def line_number(self):
        return self._lnum

    def next_token(self):
        """Return the next token to parse"""

        while True:
            
            if self.line == u'':
                self.line = self.file.readline()
                self._lnum += 1
                
            ## end of file
            if self.line == '':
                return EOF
            
            token,self.line = re.match(PAT,self.line).groups()
            if token != u'' and not re.search(r'^\;|^\#',token):
                return token

    @classmethod
    def parse(cls,file_or_stream):
        """parse an input lisp file or input stream to python list

        -- it will try to figure out input is a filename or input

        :param file_or_stream: the input file or stream to parse
        :type file_or_stream: str
        :returns: python list version of input
        :rtype: list
        """
        is_file = False
        mbuffer = None

        ## figure out if it is a file or not 
        if os.path.isfile(file_or_stream):
            is_file = True
            mbuffer = codecs.open(file_or_stream,encoding=ENCODING)
        elif re.search(LFILE,file_or_stream): #or not re.search(r'\(',file_or_stream):
            raise IOError('cannot find file: %s' % file_or_stream)
        else:
            mbuffer = ibuffer(file_or_stream)

        return cls._read(mbuffer)
        
    @classmethod 
    def _read(cls,mbuffer):
        functions_found = []
        input_port = cls(mbuffer)
        newest = input_port.next_token()

        while True:
            functions_found.append(cls._build_list(newest,input_port))
            newest = input_port.next_token()
            if newest == EOF:
                break
            
        return functions_found 

    @staticmethod
    def _build_list(token,iport):
        """a recursive function to build list datastructure

        :param token: the beginning token
        :type token: unicode
        :param iport: the file or stream buffer
        :type iport: LispReader
        """
        if u'(' == token:
            L = []
            while True:
                token = iport.next_token()
                if token == u')':
                    return L
                else:
                    L.append(LispReader._build_list(token,iport))
        elif u')' == token:
            raise SyntaxError('unepexted ) near line %d' % iport.line_number)

        elif token in quotes:
            next = iport.next_token()
            return [quotes[token],LispReader._build_list(next,iport)]
                
        elif token == EOF:
            raise SyntaxError('unexpected EOF in list near line %d' % iport.line_number)
        else:
            return make_atomic(token)
