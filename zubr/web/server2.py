## a simple server to doing stuff


import wsgiref.simple_server

query_page = """
<html>
<head>
<title>QUEST Query Integration Server</title>
<script type="text/javascript" language="javascript" src="https://ajax.googleapis.com/ajax/libs/prototype/1.7.0.0/prototype.js"></script>
<script type="text/javascript" language="javascript">

function runquery () {
    var chooser = $("historychooser"); 
    var query_text = $("querytypein").value;
    var input_chooser = $("par");
    var action_chooser = $("ach");
    var action = action_chooser.options[action_chooser.selectedIndex].value;
    var input = input_chooser.options[input_chooser.selectedIndex].value; 

    //alert("action selected is " + action);
    var newoption = document.createElement("option");
    newoption.text = query_text;
    $("historychooser").add(newoption, null);    
    var request = new Ajax.Request('/query?debug=true&query=' + encodeURIComponent(query_text) + '&actionchooser=' + encodeURIComponent(action) + '&inputchooser=' + encodeURIComponent(input), {
        method: 'get',
        onSuccess: function(response) {
            $("queryresult").innerHTML = response.responseText;
            $("queryresult").style.visibility = "visible";
            $("querytypein").value = "";
            chooser.options[0].text = "(select a previously-issued query)";             
        },
        onFailure: function(response) {
            alert("server returned error " + response.responseText + " for query " + query_text);
        },
     });
    return false;
}
function hideresults () {
    $("queryresult").style.visibility = "hidden";
}
function selectthis (v) {
    // put this query in the textarea
    var chooser = $("historychooser");
    var choice = chooser.options[chooser.selectedIndex].text;
    if (choice.charAt(0) == "(") {
        $("querytypein").value = "";
        chooser.options[0].text = "(select a previously-issued query)";
    } else {
        $("querytypein").value = choice;
        chooser.options[0].text = "(clear the query type-in window)";
    }
}
</script>
<style>
body {
    color: %(parc-blue)s;
    text-color: %(parc-blue)s;
    background-color: %(parc-light-gray)s;
    }
div#queryresult {
    color: %(parc-dark-blue)s;
    background-color: lightyellow;
    width: 100%%;
    min-height: 100px;
    white-space: pre;
    visibility: hidden;
    }
textarea#querytypein {
    color: %(parc-blue)s;
    vertical-align: middle;
    text-align: center;
    width: 100%%;
    }
</style>
</head>
<body>
<form method=GET action="/query" id="queryform" onSubmit="return runquery()">
<table width="100%%">
<tr><td colspan=3 width="40%%" align=center><h1>SAP Quest</h1></td></tr>
</table>
<p align=center>
<textarea name="query" value="" id="querytypein" placeholder="(type query here and press the 'Submit' button or select 'voice' and start talking)" rows=5 cols=80 onkeypress="hideresults();">
</textarea>
<br>
"""

query_page2 = """<input type=submit name="submit" value="Submit/Speak"> 
</form>
</form>
<div id="queryresult">
</div>
</body>
</html>
"""

hoogle = """<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head profile="http://a9.com/-/spec/opensearch/1.1/">
        <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
        <title>  Hoogle</title>
        <link type="text/css" rel="stylesheet" href="res/hoogle.css?version=4%2e2%2e26" />
        <link type="image/png" rel="icon" href="res/favicon.png" />
		<link type="image/png" rel="apple-touch-icon" href="res/favicon57.png" />
        <link type="application/opensearchdescription+xml" rel="search" href="res/search.xml" title="Hoogle" />
        <script type="text/javascript" src="res/jquery.js?version=4%2e2%2e26"> </script>
        <script type="text/javascript" src="res/jquery-cookie.js?version=4%2e2%2e26"> </script>
        <script type="text/javascript" src="res/hoogle.js?version=4%2e2%2e26"> </script>
    </head>
    <body>
<div id="links">
    <ul id="top-menu">
        <li id="instant" style="display:none;">
            <a href="javascript:setInstant()">Instant is <span id="instantVal">off</span></a>
        </li>
        <li id="plugin" style="display:none;"><a href="javascript:searchPlugin()">Search plugin</a></li>
        <li><a href="https://github.com/ndmitchell/hoogle/blob/master/README.md">Manual</a></li>
        <li><a href="http://www.haskell.org/">haskell.org</a>
</li>
    </ul>
</div>
<form action="." method="get" id="search">
    <a id="logo" href="http://haskell.org/hoogle/">
        <img src="res/hoogle.png" width="160" height="58" alt="Hoogle"
    /></a>
    <input name="hoogle" id="hoogle" class="HOOGLE_REAL" type="text" autocomplete="off" accesskey="1" value="" />
    <input id="submit" type="submit" value="Search" />
</form>
<div id="body">
<h1><b>Welcome to Hoogle</b></h1>
<ul id="left">
<li><b>Links</b></li>
<li><a href="http://haskell.org/">Haskell.org</a></li>
<li><a href="http://hackage.haskell.org/">Hackage</a></li>
<li><a href="http://www.haskell.org/ghc/docs/latest/html/users_guide/">GHC Manual</a></li>
<li><a href="http://www.haskell.org/ghc/docs/latest/html/libraries/">Libraries</a></li>
</ul>
<p>
    Hoogle is a Haskell API search engine, which allows you to search many standard Haskell libraries
    by either function name, or by approximate type signature.
</p>
<p id="example">
    Example searches:<br/>
     <a href="?hoogle=map">map</a>
<br/>
     <a href="?hoogle=%28a+-%3e+b%29+-%3e+%5ba%5d+-%3e+%5bb%5d">(a -&gt; b) -&gt; [a] -&gt; [b]</a>
<br/>
     <a href="?hoogle=Ord+a+%3d%3e+%5ba%5d+-%3e+%5ba%5d">Ord a =&gt; [a] -&gt; [a]</a>
<br/>
     <a href="?hoogle=Data%2eMap%2einsert">Data.Map.insert</a>
<br/>
	<br/>Enter your own search at the top of the page.
</p>
<p>
    The <a href="http://www.haskell.org/haskellwiki/Hoogle">Hoogle manual</a> contains more details,
    including further details on search queries, how to install Hoogle as a command line application
    and how to integrate Hoogle with Firefox/Emacs/Vim etc.
</p>
<p>
    I am very interested in any feedback you may have. Please
    <a href="http://community.haskell.org/~ndm/contact/">email me</a>, or add an entry to my
    <a href="http://code.google.com/p/ndmitchell/issues/list">bug tracker</a>.
</p>
            <div class="push"></div>
        </div>
        <div id="footer">&copy; <a href="http://community.haskell.org/~ndm/">Neil Mitchell</a> 2004-2013, version 4.2.26</div>
    </body>
</html>
"""


class WebApp:

    def __call__(self,environ,start_response):

        path = environ.get("PATH_INFO")
        method = environ.get("REQUEST_INFO")
        caller_ip = environ.get("REMOTE_ADDR")
        query = environ.get("QUERY_STRING")

        if path == "/":
            actionchooser = '<select name="actionchooser" id="ach" size=1>\n'
            actionchooser += '<option selected value="exit-lisp"> exit lisp</option>'
            actionchooser += '<option selected value="parser-output"> parser output</option>'
            actionchooser += '<option selected value="exit-system"> exit system</option>'
            actionchooser += '<option selected value="test-parc2snark"> answers and proofs</option>'
            actionchooser += '<option selected value="speech-output"> speech output</option>'
            actionchooser += '<option selected value="all-parc2snark"> all answers and proofs</option>'
            actionchooser += '<option selected value="parc-to-all-result"> all answers</option>'
            actionchooser += '<option selected value="parc-to-result"> answers</option>'
            actionchooser += '<option value="parc-to-logical-form"> Snark logical form</option>'
            actionchooser += "</select>\n"
            inputchooser = '<select name="inputchooser" id="par" size=1>\n'
            inputchooser += '<option selected value="voice"> voice</option>'
            inputchooser += '<option selected value="text"> text</option>'
            inputchooser += "</select>\n"

            historychooser = '<select id="historychooser" size=1 onchange="{selectthis(this);}">\n'
            historychooser += '<option value="*">(select a previous sentence)</option>'

            historychooser += "</select>\n"

            start_response("200 OK", [('Content-Type','text/html')])
            return [query_page,actionchooser,historychooser,query_page2]


if __name__ == "__main__":
    httpd = wsgiref.simple_server.make_server('',8080,WebApp())
    httpd.serve_forever()
        
