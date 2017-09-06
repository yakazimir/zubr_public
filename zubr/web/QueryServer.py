#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Objects for running an HTTP query server

"""

import os
import sys
import time
import traceback
import webbrowser
import cgi
import shutil
import codecs
import wsgiref.simple_server
import base64
from SocketServer import ThreadingMixIn
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler 
import urlparse
import re
import string
import logging
import datetime
from zubr.util import ConfigObj,ConfigAttrs
from zubr.QueryInterface import RerankerDecoderInterface ## should be higher up here 

MAIN_PAGE = """ 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <!-- This file has been downloaded from Bootsnipp.com. Enjoy! -->
    <title>Function Assistant</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="http://netdna.bootstrapcdn.com/bootstrap/3.1.0/css/bootstrap.min.css" rel="stylesheet">
    <style type="text/css">
        body {
           margin-top:20px;
        }
        div#page {
         	margin:auto;
            text-align:left;
            //padding:100px 100px 30px;
        }
        div#queryresult {
           color: #2F4F4F;
           background-color: #FFFFFF;
           text-align:left;
           align:center;
           width:500px;
	       margin:50px auto 0;
        }
        h1{
           font-family:Corbel,'Myriad Pro',Arial, Helvetica, sans-serif;
           background:url('Codeoogle-header.png') no-repeat center top;
           text-indent:-9999px;
           overflow:hidden;
           height:60px;
         }
       table{ 
         margin-left: auto;
         margin-right: auto;
         font-family:Corbel,'Myriad Pro',Arial, Helvetica, sans-serif;
       }

    </style>
    <script src="http://code.jquery.com/jquery-1.11.1.min.js"></script>
    <script src="http://netdna.bootstrapcdn.com/bootstrap/3.1.0/js/bootstrap.min.js"></script>

<script type="text/javascript">

$(document).ready(function(e){
    $('.search-panel .dropdown-menu').find('a').click(function(e) {
	e.preventDefault();
	var param = $(this).attr("href").replace("#","");
	var concept = $(this).text();
	$('.search-panel span#search_concept').text(concept);
	$('.input-group #search_param').val(param);
    $('.search-panel span#search_concept').text(concept);
    $('.input-group #search_param').val(param);
	});
});

function about() {
   var httpRequest = new XMLHttpRequest();
   httpRequest.open('GET','/about');
   httpRequest.send();
   $("#queryresult").html("<p><i>Function Assistant</i> is a <b>prototype query engine</b> designed to help end-users of various software libraries or APIs find information about functions using natural language. Users can provide <i>Function Assistant</i> with a description of some target functionality, and the tool will return a set of candidate functions that it thinks relate to your description.</p><p>For example, by selecting the <i>nltk</i> option in the left drop-down box (which will allow you to query a well-known NLP library called <b>NLTK</b>), you can type: <i> Train a probabilistic sequence tagger </i>, which should return functions related to training sequence models. See below for a screen demo.</p><iframe src='https://player.vimeo.com/video/218164282' width='500' height='250' frameborder='0' webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe><p><a href='https://vimeo.com/218164282'>function_assistant_demo</a> on <a href='https://vimeo.com'>Vimeo</a>.</p> <p>Please note that the tool is only a rough prototype. The provided models are, for the moment, taken directly various academic experiments, and are probably not optimized for querying. The tool is currently under heavy development. The full source code will be released soon.</p><p>For more information, see our work: <a href='https://arxiv.org/abs/1705.04815' target='_blank'> technical background, </a><a href='https://arxiv.org/abs/1706.00468' target='_blank'>demo</a>. If you have any questions or comments, please contact: zubr-support@ims.uni-stuttgart.de.</p><table width='100%%' style='background-color: #FFFFFF'><tr valign='top'><td style='background-color: #FFFFFF'><font size='-2'>[<a href='http://www.ims.uni-stuttgart.de/index.en.html' target='_blank'>IMS, University of Stuttgart</a>][<a href='http://www.ims.uni-stuttgart.de/impressum/index.en.html' target='_blank'>Legal notice</a>][<a href='http://www.ims.uni-stuttgart.de/institut/mitarbeiter/kyle/index.en.html' target='_blank'>Kyle Richardson</a> (author)]</font></td><td align='right'  style='background-color: #FFFFFF'><img src='http://www.ims.uni-stuttgart.de/img/logo_blue_left.png' width='100'/></td></tr></table>");
};

function faq() {
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET','/faq');
    httpRequest.send();
    $("#queryresult").html("<b>Q1: What is Function Assistant?</b><p>Function Assistant is tool for querying software libraries using using natural language. Users can enter a query to some software library (listed in the pull down menu), and it will attempt to match your query with functions in that library.<br><br>Links in the output representations will lead you to the original source code.</p><br><b>Q2: What is an example query?</b><p>One example (to the sklearn libary, a machine learning libary) is <i>Fit a logisitic regression model using data.</i> The model should then return functions related to training logistic regression models. <br><br> In general, the model is designed to handle unconstrained natural language input, so slight variations of this query should still yield the same or similar results.</p><b>Q3: How does Function Assistant work?</b><p>Function Assistant uses a statistical translation model to translate user queries to formal representations in the target software library. The translation model is learned by reading example software libraries and text documentation/function pairs. <br><br>For more details, please see our work: <a href='https://arxiv.org/abs/1705.04815' target='_blank'> technical background, </a><a href='https://arxiv.org/abs/1706.00468' target='_blank'>demo</a>.</p><b>Q4: Why are the results so bad?</b><p>For now, the provided models are taken directly from various academic experiment and are optimized to solve different problems. In addition, the current models are relatively simple, and are not meant to provide an end-all solution to the problem of software question-answering, but rather a baseline for future work. We are continuing to develop new models, please stay tuned.<br><br>The quality of the translation also depends on the quality and size of the documentation datasets used to train the models. Not all documentation sets are well-documented, which will therefore lead to bad translation models, and some are very small. We are actively working on these issues.</p><b>Q5: Why are some functions missing?</b><p>You might find that some functions are missing in the output of the model. Currently, the model searches within the space of all <i>documented</i> functions for a given API. In other words, if a function is not documented in the original source code, it won't appear in the output.</p><b>Q6: Why are some links broken?</b><p>Some of the function links either point to the wrong place in the target source code, or are broken. The first reason is that the links sometimes contain mistakes. The second reason is that many of the projects are under active development, and functions sometimes change position or are deleted.</p><b>Q7: Can I train my own models?</b><p>Yes! The underlying software library allows you to build new datasets and query servers from raw source code collections. The tool can also be customized using our library's internal Python/Cython API. All source code and data will be released soon, please stayed tuned.</p><b>Q8: Does Function Assistant work in other languages? </b><p>Yes! The underlying translation mechanism is agnostic to the type of input language, assuming that training data exists for the desired input language. We will put up some of these non-English models, which will appear in the drop down menu as (type-of-data)_(language_id).</p><b>Q9: How did you choose the projects featured here?</b><p>We first started with programming language standard library documentation, and chose languages according to the quality and size of their documentation. Later we looked at open source Python projects, and used the <a href='https://github.com/vinta/awesome-python' target='_blank'> Awesome-python </a> list as a guide. To date, we have experimented with around 45 APIs (involving 10 natural languages and 11 programming languages), and we hope that this number will increase in the future.<br><br>One current limitation is that our methods only work well on medium and large APIs, i.e., ones containing thousands of function annotations. There are many projects in the Awesome-python list that we would like to include here, but the number of documented functions is just too small. We are currently working on new models that avoid this limitation.<br><br>If you have a request for a particular API to be featured here, please write zubr-support@ims.uni-stuttgart.de.</p>");
};

function contact() {
      $("#queryresult").html("Please contact zubr-support@ims.uni-stuttgart.de for more information.")
};

function hideresults () {
    // hide the query result field
    $("#queryresult").html("");
};

</script>

</head>
<body>
<div id="page">
<h1>Function Assistant Tool</h1>
<table style="height: 34px; margin-left: auto; margin-right: auto;" width="480">
<tbody>
<tr>
<td>&nbsp;</td>
<td>Query code using Natural Language&emsp;</td>
<td><a href="#about" style="text-decoration:none" onclick="about();" class="button">About&emsp;</a></td>
<td><a href="#faq" style="text-decoration:none" onclick="faq();" class="button">FAQ&emsp;</a></td> 
<td><a href="#faq" style="text-decoration:none" onclick="contact();" class="button">Contact&emsp;</a></td>
<td><a href="https://github.com/yakazimir/Code-Datasets" style="text-decoration:none" class="button">Resources</a></td>
</tr>
</tbody>
</table>
<form method=GET action="/query" id="queryform" onKeyPress="hideresults();">
<div class="container" id="mainclass">
    <div class="row">    
        <div class="col-xs-8 col-xs-offset-2">
		    <div class="input-group" id="ingroup">
                <div class="input-group-btn search-panel">
                    <button type="button" class="btn btn-default dropdown-toggle" data-toggle="dropdown" onKeyPress="hideresults();">
                    	<span id="search_concept">%s</span> <span class="caret"></span>
                    </button>
                    <ul class="dropdown-menu" role="menu">
                      %s
                    </ul>
                </div>
                <input type="hidden" name="search_param" id="search_param">         
                <input type="text"  class="form-control" name="x" placeholder="Search for a function...">
                <span class="input-group-btn">s
                    <button class="btn btn-default" type="submit"><span class="glyphicon glyphicon-search"></span></button>
                </span>
            </div>
        </div>
	</div>
</div>
</form>
<div id=queryresult>
%s
</div>
</div>
</body>
</html>
"""

FUN_A="""
<i>Function Assistant</i> is a <b>prototype query engine</b> designed to help end-users of various software libraries or APIs find information about functions using natural language. Users can provide <i>Function Assistant</i> with a description of some target functionality, and the tool will return a set of candidate functions that it thinks relate to your description. <br><br>

For example, by selecting the <i>nltk</i> option (which will allow you to query a well-known NLP library called <b>NLTK</b>), you can type: <i> Train a probabilistic sequence tagger </i>, which should return functions related to training sequence models.  <br><br>

Please note that the tool is only a rough prototype. The provided models are, for the moment, taken directly various academic experiments, and are probably not optimized for querying. The tool is currently under heavy development. The full source code will be released soon. <br><br>

For more information, see our work: <a href="https://arxiv.org/abs/1705.04815" target="_blank">technical background, </a> <a href="http://www.ims.uni-stuttgart.de/institut/mitarbeiter/kyle/demo_paper.pdf" target="_blank">demo paper</a>, If you have any questions or comments, please contact: zubr-support@ims.uni-stuttgart.de. <br><br>

<table width="100%" style="background-color: #FFFFFF">
  <tr valign="top">
    <td style="background-color: #FFFFFF">
      <font size="-2">
	[<a href="http://www.ims.uni-stuttgart.de/index.en.html">IMS, University of Stuttgart</a>][<a href="http://www.ims.uni-stuttgart.de/impressum/index.en.html">Legal notice</a>][<a href="http://www.ims.uni-stuttgart.de/institut/mitarbeiter/kyle">Kyle Richardson</a> (author)]
      </font>
    </td>
    <td align="right"  style="background-color: #FFFFFF">
      <img src="http://www.ims.uni-stuttgart.de/img/logo_blue_left.png" width="100"/>
    </td>
  </tr>
</table>

"""

QUERY="""
<b>Your query is:</b><tt><i>'%s'</i></tt> <b>processed in </b> %f <b> <tt><i>seconds</i></tt> </b> <br><br>\n

"""

PUNCTUATION  = re.compile('[%s]' % re.escape(string.punctuation))

class QueryServer(object):

    """Base class for building an HTTP based query server"""

    def __init__(self,names,models,wdir=None):
        self.names    = names
        self.models   = models
        self.previous = names[0]
        self.wdir     = wdir
        self._log     = None

        if self.wdir:
            file_out = os.path.join(wdir,'query_logger.log')
            self._log = codecs.open(file_out,'w',encoding='utf-8')

    def __call__(self,environ,start_response):
        """Main method for interacting with html/javascript"""

        path = environ.get("PATH_INFO")
        query = environ.get("QUERY_STRING")
        
        if path == "/":
            start_response("200 OK", [('Content-Type','text/html; charset=utf-8')])
            drop_down = self.__make_options()
            main_page = MAIN_PAGE % (self.previous,drop_down,"")
            return [main_page.encode('utf-8')]

        elif path == "/faq":
            start_response("200 OK", [('Content-Type','text/html; charset=utf-8')])
            self.logger.info('Returning faq information')
            drop_down = self.__make_options()
            return [MAIN_PAGE % (self.previous,drop_down,FUN_A)]
            
        elif path == "/about":
            start_response("200 OK", [('Content-Type','text/html; charset=utf-8')])
            self.logger.info('Returning about information')
            drop_down = self.__make_options()
            
            return [MAIN_PAGE % (self.previous,drop_down,FUN_A)]

        elif path == "/query":
            start_response("200 OK", [('Content-Type','text/html; charset=utf-8')])

            ## parse the incoming query 
            parsed_input = urlparse.parse_qs(query)
            query = parsed_input.get("x",None)
            language = parsed_input.get('search_param',None)
            drop_down = self.__make_options()
            ## set the current language to the last one used
            language = self.previous if not language else language[0].lower()
            self.previous = language
            ## EMPTY QUERY?
            if not query: return [MAIN_PAGE % (self.previous,drop_down,"")]

            ## take the query
            try:
                query = query[0]
                model_index = self.names.index(language)
                query_model = self.models[model_index]
                query_out = query_model.query(query,20)

                ## log the output
                self.logger.info('Received query: %s, for language: %s' % (query,language))
                ## display the query (with preprocessing)
                first_out = QUERY % (query,query_out.time)
                first_out += str(query_out.rep)

                ## log to internal file
                if self._log:
                    try: 
                        time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
                        print >>self._log,"%s\t%s\t%s\t%s" % (time,language,query,' '.join([str(i) for i in query_out.ids]))
                    except Exception,e:
                        self.logger.error('Error to print to log! ')
                        self.logger.error(e,exc_info=True)

            except Exception,e:
                self.logger.error(e,exc_info=True)
            
            ## add the query_out
            return [MAIN_PAGE % (self.previous,drop_down,str(first_out))] #str(query_out.rep))]

        elif path == "/favicon.ico":
            start_response("200 OK",[('Content-Type','text/plain; charset=utf-8')])
            return []

        ## apple specific stuff 
        elif path == '/apple-touch-icon-120x120.png HTTP/1.1':
            start_response("200 OK",[('Content-Type','text/plain; charset=utf-8')])
            return []

        elif path == "/Codeoogle-header.png":
            logo = base64.decodestring(LOGO.strip())
            start_response("200 OK",[('Content-Type','image/png; charset=utf-8')])
            return [logo,]

    def __make_options(self):
        """Make the option list for the pull down menu

        """
        return '\n'.join(["""<li><a href=#%s> %s </a></li>""" %  (i,i) for i in self.names])

    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        ## close the underlying log file 
        if self.wdir and self._log:
            self._log.close()

    @property
    def logger(self):
        """a logger object for the query server

        :returns: logger instance
        """
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level)

class MyWSGIServer(ThreadingMixIn, wsgiref.simple_server.WSGIServer): 
     pass


LOGO="""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"""



## CLI STUFF

def params():
    """Main parameters for building a query server  

    :returns: description of configuration item and configuration options 
    """
    groups = {"QueryServer" : "Settings for query servers"}
    
    options = [
        ("--port","port",5000,"int",
         "The port to run server on [default=5000]","QueryInterface"),
        ("--open_browser","open_browser",True,"bool",
         "Open the browser by default [default='']","QueryInterface"),
        ("--qmodels","qmodels",'',"str",
         "The query models to load [default='']","QueryInterface"),
        ("--out_size","out_size",20,"int",
         "The size of the output [default=20]","QueryInterface"),
    ]

    return (groups,options)

def argparser():
    """Returns the query interface configuration argument parser 

    :rtype: zubr.util.config.ConfigObj 
    :returns: default argument parser 
    """
    from zubr import _heading
    from _version import __version__ as v

    usage = """python -m zubr queryserver [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

def main(argv):
    """The main execution point for the running a query server 

    :param argv: the cli or configuration input settings 
    """
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])
        logging.basicConfig(level=logging.DEBUG)

    server_loader = logging.getLogger('zubr.web.QueryServer')

    names  = []
    models = []

    try: 
        ## by default, the name of directory where
        ## model is should be the name of the project 
        for project in config.qmodels.split("+"):
            pname = os.path.basename(os.path.dirname(project))
            names.append(pname.lower())
            rlist = os.path.join(os.path.dirname(project),"rank_list.txt")
            if os.path.isfile(rlist) and config.dir:
                copied_rlist = os.path.join(config.dir,"rank_list_%s.txt" % pname.lower())
                shutil.copy(rlist,copied_rlist)
            
            server_loader.info('Building model for %s, might take a while..' % pname)
            try: 
                model = RerankerDecoderInterface.load(project)
            except Exception,e:
                server_loader.error('Error building model:%s, skipping' % project)
                server_loader.error(e,exc_info=True)
                
            server_loader.info('Finished building the model') 
            models.append(model)

        port = 5000 if not config.port else config.port

        with QueryServer(names,models,config.dir) as query_app:

            try: 
                server = wsgiref.simple_server.make_server('',port,
                                                               query_app,
                                                               MyWSGIServer,
                                                               wsgiref.simple_server.WSGIRequestHandler)
                if config.open_browser:
                    webbrowser.open('http://localhost:%d' % port,new=2,autoraise=True)
                server.serve_forever()
            except KeyboardInterrupt:
                print "^C received, shutting down the web server"
                try:
                    query_app.logger.info('Shutting down!')
                except:
                    pass
                server.socket.close()
            
    except Exception,e:
        traceback.print_exc(file=sys.stdout)
        server_loader.error(e,exc_info=True)

if __name__ == "__main__":
    main(sys.argv[1:])
