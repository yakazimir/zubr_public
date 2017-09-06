# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""
from shutil import copytree,copy,rmtree
import os
import sys
import time
import logging
from zubr import _heading
from zubr._version import __version__ as v
from zubr.util import ConfigObj
from zubr.Query import Query
from zubr.web.server import QueryApp
import wsgiref.simple_server
import webbrowser


def params():
    """specifies the code-server specific settings

    :returns: code-server options plus a description
    :rtype: tuple
    """
    from zubr.Aligner import params
    aligner_group,aligner_param = params()
    aligner_group["CodeServer"] = "Settings for code query server"    

    options = [
        ("--name","name",'generic',"str",
         "name of query object [default='generic']","ServerOptions"),
        ("--query_type","query_type","aligner","str",
         "type of query model [default=aligner]","ServerOptions"),
        ("--port","port",7000,"int",
         "port to put server [default=7000]","ServerOptions"),
        ("--model_file","model_file","",str,
         "location of model file [default=""]","ServerOptions"),
        ("--data_path","data_path",'','str',
         "location of the pre-built data (if exists) [default='']","QuerEr"),
        ("--number_to_show","number_to_show",10,int,
         "number of items to display [default=""]","ServerOptions"),
        ("--open_browser","open_browser",True,"bool",
         "automatical open a browser [default='True']","ServerOptions"),
    ]
        
    options += aligner_param
    return (aligner_group,options)

def main(config):
    """main method for running code query server

    :param config: main configuration object
    :rtype: None 
    """
    mod_map = {}
    lang_map= {}
    
    if isinstance(config.name,list):
        rank_files = config.rfile
        names      = config.name
        train_loc  = config.atraining
        num_models = len(rank_files)

        for i in range(num_models):
            config.rfile     = rank_files[i]
            config.name      = names[i]
            config.atraining = train_loc[i]
            ## rank files
            main_rank = os.path.join(config.dir,"rank_list.txt")
            main_uri  = os.path.join(config.dir,"rank_list_uri.txt")
            lang_rank = os.path.join(config.dir,"rank_list_%s.txt" % config.name)
            lang_uri  = os.path.join(config.dir,"rank_list_uri_%s.txt" % config.name)
            copy(lang_rank,main_rank)
            copy(lang_uri,main_uri)
            ## query object
            query_obj = Query(config)
            #query_obj._logger('loaded model: %s' % config.name)
            mod_map[config.name] = query_obj
            lang_map[config.name]= config.name
            ## remove rankfile, alignment directory
            alignmentdir = os.path.join(config.dir,"alignment")
            rmtree(alignmentdir)
            os.remove(main_rank)
            os.remove(main_uri)
            
    else:
        query_obj = Query(config)
        query_map[config.name] = query_obj
        lang_map[config.name]  = config.name 

    app = QueryApp(model=mod_map,lang_ops=lang_map)
            
    try:
        server = wsgiref.simple_server.make_server('',config.port,app)
        #if config.open_browser:
        #    webbrowser.open('http://localhost:%s' % config.port,new=2,autoraise=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print '^C received, shutting down the web server'
        server.socket.close()

if __name__ == "__main__":
    main(sys.argv[1:]) 
