

# import any other python module in the normal fashion

## script parameters
## format: cli switch, switch name, default val, type, description, script name
params = [
    ("--param1","param1",False,"bool","description of param1","MyScript"),
    ("--param2","param2",False,"bool","description of param2","MyScript")
]

## name and description of script ( script name -> description)
description = {"MyScript":"settings for my script"}

## tasks to perform in pipeline (in order)
## anything with zubr.X will run module X in zubr toolkit
tasks = [
    "setup_pipeline",
    "zubr.Alignment",
    "close_pipeline",
]

## all script functions should have config as first argument (regardless of
## whether they use a config or not)

def setup_pipeline(config):
    ## function at beginning of script/pipeline, e.g., building data 
    print "setting up pipeline, building data"

def close_pipeline(config):
    ## function for end of script/pipeline, e.g., evaluating results
    print "closing the pipeline, evaluate results,..."

