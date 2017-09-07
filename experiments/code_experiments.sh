# This script tells you how to get the data and associated pipelines for
# reproducing the following two publications: 
#
#
# @inproceedings{richardson-kuhn:2017:Long,
# author    = {Richardson, Kyle  and  Kuhn, Jonas},
# title     = {Learning {S}emantic {C}orrespondences in {T}echnical {D}ocumentation},
# booktitle = {Proceedings of the ACL},
# year      = {2017},
# url={http://aclweb.org/anthology/P/P17/P17-1148.pdf},
#   }
#
# @inproceedings{richardson-kuhn:2017:Demo,
# author    = {Richardson, Kyle  and  Kuhn, Jonas},
# title     = {Function {A}ssistant: {A} {T}ool for {NL} {Q}uerying of {API}s},
# booktitle = {Proceedings of the EMNLP},
# year      = {2017},
#   }
#
# If you have any problems reproducing the results, or setting up the infrastructure,
# please write an email to kyle@ims.uni-stuttgart.de

## change directory to top
SCRIPT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ZUBR_LOC="$( dirname $SCRIPT_LOC )"
TECH_DOC=$ZUBR_LOC/experiments/technical_documentation
echo "...MOVING UP TO ZUBR SRC..."
cd $ZUBR_LOC

## setup the technical documentation dir

if [ ! -d "$TECH_DOC" ] ; then
    echo "...BUILDING EXPERIMENT DIRECTORY..."
    mkdir $TECH_DOC
fi

### GETTTING THE CODE DATA: the data is hosted here for both datasets:
### https://github.com/yakazimir/Code-Datasets

echo "...DOWNLOADING THE DATA...."

## download the pipeline and associated scripts


