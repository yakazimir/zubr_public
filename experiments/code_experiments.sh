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
echo "...MOVING UP TO ZUBR SRC DIR..."
cd $ZUBR_LOC

## setup the technical documentation dir

if [ ! -d "$TECH_DOC" ] ; then
    echo "...BUILDING EXPERIMENT DIRECTORY..."
    mkdir $TECH_DOC
fi

## check if the data already exists, if so break

if [ -d "$TECH_DOC/data" ] ; then
    echo "DATA ALREADY EXISTS! Please remove (exiting...)"
    exit 
fi

## check to continue

read -p "This is a lot of data, Continue (Y/n)? " choice
case "$choice" in 
  Y ) echo "ACCEPTED PROMPT";;
  n|N ) exit;;
  * ) exit;;
esac

### GETTTING THE CODE DATA: the data is hosted here for both datasets:
### https://github.com/yakazimir/Code-Datasets
echo "...DOWNLOADING THE DATA AND SCRIPTS...."
wget https://github.com/yakazimir/Code-Datasets/archive/master.zip -O $TECH_DOC/data.zip

## unzip the data
## unzip the zip inside the downloaded data
echo "...UNZIPPING THE DATA FILE..."
unzip $TECH_DOC/data.zip -d $TECH_DOC
echo "...MOVING DATA UP..."
## remove the bin (don't need it)
mv -f experiments/technical_documentation/Code-Datasets-master/*/ experiments/technical_documentation/

## unzip the acl_emnlp files
echo "...UNZIPPING THE PIPELINE DATA..."
unzip experiments/technical_documentation/Code-Datasets-master/acl_emnlp_naacl.zip -d $TECH_DOC
echo "...REMOVING AUXILIARY BIN FILE (NOT NEEDED)..."
rm -rf $TECH_DOC/bin
mv -f experiments/technical_documentation/acl_emnlp_naacl/* experiments/technical_documentation
echo "...REMOVING THE AUXILIARY ACL_EMNLP_NAACL FILE..."
rm -rf experiments/technical_documentation/acl_emnlp_naacl

##unzip the semantic parsing data
echo "...UNZIPPING THE SEMANTIC PARSING (SP) DATA..."

read -p "Unzip the extra Semantic Parsing data (not needed if just running code experiments)? (Y/n)" choice
case "$choice" in
    Y ) unzip $TECH_DOC/other_data/SP -d $TECH_DOC/other_data;;
    n|N ) ;;
    * ) ;;
esac 

## make the run directory
echo "...MAKING A RUN DIRECTORY"
mkdir $TECH_DOC/runs

echo "FINISHED: please see REPRODUCE.txt for more information"
