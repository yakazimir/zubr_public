#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

A simple server app for running code search

"""
import os
import sys
import time
import traceback
import webbrowser
import cgi
import wsgiref.simple_server
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep
import cgi

PORT_NUMBER = 7000

class myHandler(BaseHTTPRequestHandler):
    
    #Handler for the GET requests
    def do_GET(self):
        if self.path=="/":
            self.path="/search.html"

        try:
            #Check the file extension required and
            #set the right mime type

            sendReply = False
            if self.path.endswith(".html"):
                mimetype='text/html'
                sendReply = True
            if self.path.endswith(".png"):
                mimetype='img/png'
                sendReply = True
            if self.path.endswith(".gif"):
                mimetype='img/gif'
                sendReply = True
            if self.path.endswith(".js"):
                mimetype='application/javascript'
                sendReply = True
            if self.path.endswith(".css"):
                mimetype='text/css'
                sendReply = True

            if sendReply == True:
                f = open(curdir + sep + self.path) 
                self.send_response(200)
                self.send_header('Content-type',mimetype)
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                
            return

        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)

    def do_POST(self):
        return 
    
        # self.send_response(200)
        # self.end_headers()

        # form = cgi.FieldStorage(
        #     fp=self.rfile, 
        #     headers=self.headers,
        #     environ={'REQUEST_METHOD':'POST',
        #     'CONTENT_TYPE':self.headers['Content-Type'],
        #   })
        
        # self.wfile.write("Thanks %s !" % form["your_name"].value)
        
    #Handler for the POST requests
    # def do_POST(self):
    #     print "yes"
    #   if self.path=="/send":
    #       form = cgi.FieldStorage(
    #           fp=self.rfile, 
    #           headers=self.headers,
    #           environ={'REQUEST_METHOD':'POST',
    #                    'CONTENT_TYPE':self.headers['Content-Type'],
    #       })

    #       print "Your name is: %s" % form["your_name"].value
    #       self.send_response(200)
    #       self.end_headers()
    #       self.wfile.write("Thanks %s !" % form["your_name"].value)
    #       return      


if __name__ == "__main__":
            
    try:
        server = HTTPServer(('', PORT_NUMBER), myHandler)
        webbrowser.open('http://localhost:%s' % PORT_NUMBER,new=2,autoraise=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print '^C received, shutting down the web server'
        server.socket.close()
