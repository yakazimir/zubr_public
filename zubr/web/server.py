#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

A simple server app for running an HTTP server


"""
import os
import sys
import time
import traceback
import webbrowser
import cgi
import wsgiref.simple_server
import base64
import urlparse
import re
import string
import logging

ENCODING     ='utf-8'
PORT_NUMBER  = 7000
CURDIR       = os.path.abspath(os.path.dirname(__file__))
D_STYLES     = os.path.join(CURDIR,"styles.css")
D_JAVS       = os.path.join(CURDIR,"functions.js")
PUNCTUATION  = re.compile('[%s]' % re.escape(string.punctuation))

MAIN_PAGE = """
<html>
<head>
<meta charset="UTF-8">
<title>Function Assistant</title>
<script type="text/javascript" language="javascript" src="https://ajax.googleapis.com/ajax/libs/prototype/1.7.0.0/prototype.js"></script>
<script type="text/javascript" language="javascript">%s</script>
<style>%s</style>
</head>
<body>
<div id="page">
<h1>Code-oogle API Search</h1>
<form method=GET action="/query" id="queryform" onSubmit="return run_query()">
<p align=center>
<textarea name="query" value="" id="querytypein" placeholder="(search for a function in an API documentation collection)" rows=2 cols=100 onkeypress="hideresults();">
</textarea>
<br>
<br>
"""

RESULT_PAGE = """<input type=button id="generate" onclick="return genquery()" name="Generator" value="Generate">
<input type=submit id="searchBar" name="Submit1" value="Search">
</form>
<div id="queryresult">
</div>
</div>
</body>
</html>
"""

__all__ = ['QueryApp','Server']

class QueryApp:

    """Class for building http query servers"""

    def __init__(self,model,lang_ops,styles=D_STYLES,jscript=D_JAVS):
        """initialize a qeuryapp object

        :param model: base model for generating output
        :param lang_ops: languages to load in search
        """
        self.model = model
        self.lang_ops = lang_ops

        ## page details
        self._styles = styles
        self._jscript = jscript

    def _get_styles(self):
        """opens the css file and returns values

        :returns: the string rep of css file
        :rtype: str
        """
        style_spec = ''
        with open(self._styles) as my_styles:
            style_spec = my_styles.read()
        return style_spec

    def _get_javascript(self):
        """opens the file containing the javascript functions


        :returns: the string of javascript function
        :rtype: str
        """
        j_script = ''
        with open(self._jscript) as my_jscript:
            j_script = my_jscript.read()
        return j_script

    def __call__(self,environ,start_response):
        """main method for interacting with html/javascript"""

        path = environ.get("PATH_INFO")
        query = environ.get("QUERY_STRING")

        ## load the page 
        if path == "/":

            start_response("200 OK", [('Content-Type','text/html')])
            rsize   = self.display_rank()
            lang    = self.langops()
            styles  = self._get_styles()
            jscript = self._get_javascript() 
            
            main_page = MAIN_PAGE % (jscript,styles)             
            return [main_page,lang,rsize,RESULT_PAGE]

        elif path == "/query":
            start_response("200 OK", [('Content-Type','text/plain')])
            parsed_input = urlparse.parse_qs(query)        
            if 'query' not in parsed_input:
                return [""] 
            
            lang = parsed_input['actionchooser'][0]
            rsize = int(parsed_input['rsize'][0])
            text_query = self.process_query(parsed_input['query'])
            self._logger.info('new query: %s' % text_query)
            model = self.model[lang]

            ## process the query 
            output = model.query(text_query,rsize) 
            new_output = [] 

            for i in range(0,rsize):
                new_output += [str(output[i])]
                self._logger.info('\t response %d: %s' % (i,str(output[i])))
                
            return new_output

        elif path == "/favicon.ico":
            start_response("200 OK",[('Content-Type','text/plain')])
            return []

        elif path == "/Codeoogle-header.png":
            logo = base64.decodestring(LOGO.strip())
            start_response("200 OK",[('Content-Type','image/png')])
            return [logo,]

        elif path == "/generate":
            start_response("200 OK",[('Content-Type','text/plain')])
            parsed_input = urlparse.parse_qs(query)
            lang = parsed_input['actionchooser'][0]
            model = self.model[lang]
            
            random = model.random_text()
            return [random,]

    def langops(self):
        """determines the languages to search

        :returns: language selector html string
        :rtype: str
        """
        lang = '<select name="lang" id="ach" size=1>\n'
        ## add each language 
        for name,descr in self.lang_ops.items():
            lang += '<option selected value="%s">%s</option>' % (name,descr)
        lang += "</select>\n"
        return lang

    def display_rank(self,rank=50):
        """generate the html for choosing rank

        :returns: html string
        :rtype: str
        """
        rsize = '<select name="inputchooser" id="rsize" size=1>\n'
        rsize += '<option value="1"> Top 1</option>'
        rsize += '<option selected value="10"> Top 10</option>'
        rsize += '<option value="15"> Top 15</option>'            
        rsize += '<option value="20"> Top 20</option>'
        rsize += '<option value="30"> Top 30</option>'
        rsize += '<option value="40"> Top 40</option>'
        rsize += '<option value="50"> Top 50</option>'                        
        rsize += "</select>\n"
        return rsize
 
    @property
    def _logger(self):
        """a logger object for the query server

        :returns: logger instance
        """
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level) 

    def process_query(self,query):
        """some simple pre-processing for each query

        :param query: the query to process
        """
        text = re.sub('[%s]' % re.escape(string.punctuation),'',query[0])
        text = re.sub(r'\n+','',text).lower()
        return text


### server factory

class Server(object):

    """Factory class for assinging a server object"""

    def __new__(self,name):
        """returns a new server object using ``name''

        :param name: name of server types
        """
        name = name.lower()
        
        if name in ['query','queryapp']: ## only one supported at the moment
            return QueryApp
        raise ValueError('unknown server type: %s' % name)
            

########## CLI STUFF
#####################
#####################

LOGO="""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"""

