# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Utility for representation machine learning features 
"""

import logging
import mmap
import os
import gzip
from collections import defaultdict
import numpy as np
cimport numpy as np
from zubr.DefaultMap cimport DefaultIntegerMap
from zubr.ZubrClass cimport ZubrSerializable
from cython cimport boundscheck,wraparound,cdivision

gen_logger = logging.getLogger('zubr.Features.gen_logger')

cdef class FeatureCounter(dict):
    """A class for counter features"""

    def __getattr__(self,x):
        try:
            return self[x]
        except (KeyError,AttributeError):
            # don't need to add it
            #self[x] = 0
            return 0

    def __setattr__(self,x,val):
        self[x] = val

cdef class TranslationCounter(FeatureCounter):
    """Class for keeping track of the translation features"""
    pass

cdef class Vectorizer:
    """Creates a vector representation for a dictionary feature pairs"""

    def __init__(self,feature_dict):
        """ 

        :param feature_dict: a dictionary representation of features 
        :type feature_dict: defaultdict
        """
        cdef int k,size
        cdef long feature_num
        cdef double feature_val
        
        size = <int>len(feature_dict)
        count_rep = np.ndarray((size,),dtype='d')
        #ident_rep = np.ndarray((size,),dtype=np.int32)
        ident_rep = np.ndarray((size,),dtype=np.long)
        
        for k,(feature_num,feature_val) in enumerate(feature_dict.iteritems()):
            #if feature_val <= 0: continue 
            count_rep[k] = feature_val
            ident_rep[k] = feature_num

        self.feat_counts = count_rep
        self.features    = ident_rep

    property size:
        
        """The size of the feature pair vector"""

        def __get__(self):
            """Returns the size of the pair vector 

            :rtype: int
            """
            return <int>self.feat_counts.shape[0]
        

## special feature map

cdef class FeatureObj(ZubrSerializable):

    def __init__(self,beam,flist=[],gfeatures=None,templates={},maximum=-1,baseline=-1):
        """ 

        :param num_f: number of features in each object 
        :param beam: number of items in rank list 
        """
        self.beam = beam

        if not flist:
            self._features = [FeatureMap(templates,maximum) for i in range(beam)]
        else: self._features = flist

        if not gfeatures: self._gold_features = FeatureMap(templates,maximum)
        else: self._gold_features = gfeatures

        ## information about baseline rank position
        self.baseline = baseline
            
    def __getitem__(self,int i):
        return <FeatureMap>self._features[i]

    cdef Vectorizer vectorize_item(self,int i):
        """Convert a given feature representation into two vector representations

        :param i: the index of the desired feature representation
        :rtype: Vectorizer
        """
        return <Vectorizer>Vectorizer(self._features[i])

    cdef Vectorizer vectorize_gold(self):
        """Convert a gold feature representation into two vector representations

        :rtype: Vectorizer
        """
        return <Vectorizer>Vectorizer(self._gold_features)
    
    ## a prety for accessing individual features
    
    property features:

        """The list of features in beam"""

        def __get__(self):
            return <list>self._features
        def __set__(self,list feat):
            self._features = feat

    property gold_features:
        """Features associated with the gold item (if exists)"""

        def __get__(self):
            return <FeatureMap>self._gold_features
        def __set__(self,FeatureMap gold):
            self._gold_features = gold

    cpdef void print_features(self,wdir,ftype,identifier,rvals,gold):
        """Print feature representations to file
        
        :param wdir: the working directory
        :param ftype: the type of features (e.g., train,test,...) 
        :param ranks: the rank ids associated with features
        :rtype: None
        """
        directory_path = os.path.join(wdir,ftype+"_features")
        file_path = os.path.join(directory_path,str(identifier))
        __print_features(file_path,self,rvals,gold)
        
    cpdef void create_binary(self,wdir,ftype,identifier):
        """Backs up feature object to a pickled object

        :param wdir: the working directory 
        :param ftype: the type of data 
        :param identifier: the universal identifier of data
        :rtype: None 
        """
        directory_path = os.path.join(wdir,ftype+"_features")
        file_path = os.path.join(directory_path,str(identifier))
        self.dump(file_path)

    ## doesnt work for some reason, avoid using 
    @classmethod
    def load_from_file(cls,wdir,ftype,identifier,maximum,baseline=-1):
        """Load a feature object from file

        :param wdir: the working directory 
        :param ftype: the type of feature set 
        :param maximum: the maximum number of features 
        :rtype: FeatureObj
        """
        if ftype == 'valid-select': ftype = 'valid'
        directory_path = os.path.join(wdir,ftype+"_features")
        file_path = os.path.join(directory_path,str(identifier))
        rep = __load_features(file_path,ftype,maximum)
        rep.baseline = baseline
        return rep

    @classmethod
    def load_from_binary(cls,wdir,ftype,identifier):
        """Load from a serialized (pickled) backup of the feature object

        :param wdir: the working directory 
        :param ftype: the type of feature set 
        :param identifier: the particular feature item to load
        :rtype: FeatureObj
        """
        directory_path = os.path.join(wdir,ftype+"_features")
        file_path = os.path.join(directory_path,str(identifier))
        return cls.load(file_path)
                
    @staticmethod
    def features_exist(wdir,ftype,identifier):
        """Check if features exists in file already
        
        :param wdir: the working directory 
        :param ftype: the type of features 
        :param identifier: the global identifier
        :rtype: bool
        :returns: true if the file exists false otherwise
        """
        if ftype == 'valid-select': ftype = 'valid'
        directory_path = os.path.join(wdir,ftype+"_features")
        file_path = os.path.join(directory_path,str(identifier))
        return os.path.isfile(file_path+'.gz')

    def __reduce__(self):
        ## pickle implementation
        #return FeatureObj,(self.beam,self._features,self._gold_features,{},-1)
        return (rebuild_feature_object,
                    (self.beam,self._features,self._gold_features))

def rebuild_feature_object(beam,features,gold_features):
    f = FeatureObj(beam)
    f.features = features
    f.gold_features = gold_features
    return f

cdef class TemplateManager(dict):

    """Stores and computes information about feature templates"""

    def __init__(self,dict input_templates):
        """ 
        
        :param input_templates: a list of feature templates with a description 
        """
        self.description = input_templates
        self.starts      = {}
        for identifier in input_templates.keys():
            self[identifier] = 0 

    def print_description(self,path):
        """Print a description of the feature templates

        :param path: the target place to print description  
        """
        __report_features(path,self.description,self,self.starts,self.num_features)

    def compute_starts(self):
        """Compute the starting positions for each feature template

        :rtype: None 
        """
        for i in range(max(self)+1):
            if self.get(i,0) <= 0: continue
            before = sum([self.get(j,0) for j in range(0,i) if j in self])
            self.starts[i] = before
        
    property num_features:
        """The numbers of features calculated from individual templaetes"""

        def __get__(self):
            return <long>sum(self.values())
            #return <int>sum(self.values())

    def __reduce__(self):
        ## pickle implementation
        return TemplateManager,(self.description,)

cdef class BinnedFeatures(DefaultIntegerMap):
    pass

EMPTY_BIN = BinnedFeatures()
EMPTY_TEMPLATE = TemplateManager({})

cdef BinnedFeatures empty_bin():
    return <BinnedFeatures>BinnedFeatures()
    
cdef class FeatureMap(DefaultIntegerMap):

    """Holds features with counts for optimization. Can also be used for adding features 
    (given constraints specified in the templates attribute), and computing feature bins. 

    """
    def __init__(self,templates,maximum):
        """Initializes a FeatureMap

        
        :param templates: the feature extractor templates 
        :param maximum: the maximum feature identifier 
        """
        self._templates   = templates
        self._binned      = empty_bin()
        self.maximum      = maximum

    ## c level function for adding features
            
    cdef int add_binary(self,int vindx,unsigned long increm, double value=1.0) except -1:
        """Add a feature to the feature map

        -- Assumes that vindx 0 is also used

        -- Checks that feature index is within acceptable range  

        :param vindx: the index of the feature template in the weight vector 
        :param increm: the amount of the increment
        :raises: ValueError 
        """
        cdef dict templates = self._templates
        cdef long template_start = templates.get(vindx,0)
        cdef long vector_position = template_start+increm
        cdef long max_size = self.maximum

        if vector_position > max_size:
            raise ValueError('Feature value exceeds maximum! %s' % vector_position)
        
        if <int>(vindx == 0) or <bint>(template_start > 0):
            self[<long>template_start+increm] = value

    cdef int add_increm_binary(self,int vindx,unsigned long increm, double value=1.0) except -1:
        """Add a feature to the feature map

        -- Assumes that vindx 0 is also used

        -- Checks that feature index is within acceptable range  

        :param vindx: the index of the feature template in the weight vector 
        :param increm: the amount of the increment
        :raises: ValueError 
        """
        cdef dict templates = self._templates
        cdef long template_start = templates.get(vindx,0)
        cdef long vector_position = template_start+increm
        cdef long max_size = self.maximum

        if vector_position > max_size:
            raise ValueError('Feature value exceeds maximum! %s' % vector_position)
        
        if <int>(vindx == 0) or <bint>(template_start > 0):
            if vector_position not in self:
                self[vector_position] = value
            elif self[vector_position] < 4.0:
                self[vector_position] += value

    cdef int add_internal(self,int vindx, unsigned long offset,unsigned long increm,double value=1.0) except -1:
        """Add to a given feature template with an offset inside of the template

        :param vindx: the feature template id 
        :param offset: the offset, or starting point, within that template
        :param increm: the increment value starting from offset
        """
        cdef dict templates = self._templates
        cdef long template_start = templates.get(vindx,0)
        cdef long vector_position = template_start+offset+increm
        cdef long max_size = self.maximum

        if vector_position > max_size:
            raise ValueError('Feature value exceeds maximum! %s' % vector_position)

        if <int>(vindx == 0) or <bint>(template_start > 0):
            self[vector_position] = value
        
    cdef int load_from_string(self,str feature_input) except -1:
        """Load features from a string representation. 

        -- The feature representation should be a single line 
        with the following format: feat1=val1 feat2=val2 ....

        :param feature_input: the string feature input 
        :rtype: None 
        """
        cdef str feature_num,feature_val,item,i
        #cdef list features = [i.split('=') for i in feature_input.strip().split()]
        cdef list features # = [i.split('=') for i in feature_input.strip().split('\t')[-1].split()]

        try: 
            features = features = [i.split('=') for \
                                    i in feature_input.strip().split('\t')[-1].split()]

            for (feature_num,feature_val) in features:
                self[long(feature_num)] = float(feature_val)

        except ValueError:
            gen_logger.warning('Error parsing feature input: %s, probably empty, skipping..' % feature_input)
            
        except Exception,e:
            #self.logger.error('Error encountered on %s\n%s' % (feature_input,e))
            gen_logger.error('Encountered error on %s' % feature_input)
            raise e
            
    cdef void add_incr(self,int vindx,unsigned long increm, double value):
        """Add a feature to the feature map

        :param vindx: the index of the feature template in the weight vector 
        :param increm: the amount of the increment
        :rtype: None 
        """
        cdef dict templates = self._templates
        cdef unsigned long template_start = templates.get(vindx,0)

        if (vindx == 0) or <bint>(template_start > 0):
            self[<long>template_start+increm] += value

    cdef void add_binned(self,int vindx):
        """Add a binned feature to feature map 

        :param vindx: the starting point of binned feature template 
        """
        cdef BinnedFeatures binned = self._binned
        cdef dict templates = self._templates

        if <bint>(<int>templates.get(vindx,0) > 0):
            binned[vindx] += 1.0
            
    cdef void compute_neg_bins(self,int[:] negative,double threshold):
        """Compute binned features and add to feature map

        -- Note: bin counts occurence of certain features according 
        to a range

        -- for example: 
               feature x occurs <= 0        : bin1 (negative feature)
               feature x occurs 0 < 1 <= 2  : bin2      
               feature x occurs 2 < 3 <= 4  : bin3 
               feature x occurs 4 < 5 <= 6  : bin4
               feature x occurs > 6         : bin5 
               ....
        
        :param negative: negative features 
        :param binsize: the size of the bins (by default 5)
        """
        cdef int i,temp_id,num_neg = negative.shape[0]
        cdef int count
        cdef BinnedFeatures binned = self._binned

        for i in range(num_neg):
            temp_id = negative[i] 
            
            if temp_id not in binned:
                self.add_binary(temp_id,0)
                continue

            count = binned[temp_id]
            if count >= 1.0:
                self.add_binary(temp_id,1)
            if count >= threshold:
                self.add_binary(temp_id,2)
            if count >= (threshold+threshold):
                self.add_binary(temp_id,3)
            if count >= (threshold+threshold+threshold):
                self.add_binary(temp_id,4)

    property feature_size:
        """The number of features being held"""
        
        def __get__(self):
            """Returns the number of features

            :rtype: int
            """
            return <int>(len(self))

    def __reduce__(self):
        ## pickle implementation
        #return FeatureMap,(self._templates,self.maximum)
        return (rebuild_feature_map,(self.keys(),self.values(),
                                         self._templates,self.maximum))

def rebuild_feature_map(keys,values,templates,maximum):
    new_map = FeatureMap(templates,maximum)
    new_map.update(dict(zip(keys,values)))
    return new_map

cdef class FeatureAnalyzer(ZubrSerializable):

    """This class keeps tracks of feature items, probabilities, etc..."""

    def __init__(self,size):
        """Initializes an empty featureanalysis 

        :param size: the size of the update beam 
        """
        self.size = size
        self.likelihood = 1e-1000
        ## first position is the gold 
        self.probs          = np.zeros(size+1,dtype='d')
        self.feature_scores = {}
        ## 
        self.averaged_nonzeroed = 0.0
        
    cdef double gold_prob(self):
        """Returns the gold probability or score

        :rtype: double
        """
        cdef double[:] probs = self.probs
        return probs[0]


    cdef void add_gold_feature(self,long identifier,double value,double score):
        """Add a gold feature to overall feature list for updates

        -- Note that gold item occuries 0 portion of probs vector 

        :param identifier: the feature identifier 
        :param val: the feature val 
        """
        cdef double[:] probs = self.probs
        cdef dict flist = self.feature_scores
        cdef int size = self.size+1
        
        if identifier not in flist:
            flist[identifier] = np.zeros((size,),dtype='d')

        flist[identifier][0] = value
        probs[0] += score

    cdef void add_feature(self,int num, long identifier,double value,double score):
        """Add a (non-gold) feature to the feature list for updates

        :param num: the location of item in beam 
        :param identifier: feature identifier 
        :param param value: the feature value for this instance 
        :param score: the feature score
        """
        cdef double[:] probs = self.probs
        cdef dict flist = self.feature_scores
        cdef int size = self.size+1

        if identifier not in flist:
            flist[identifier] = np.zeros((size,),dtype='d')

        flist[identifier][num+1] = value
        probs[num+1] += score

    @cdivision(True)
    cdef int normalize(self,int start=0,int end=0) except -1:
        """Normalize the probabilities using the raw feature counts and compute (log) likelihood 

        -- Note: the likelihood is not computed with regularization terms, so it
        might actually deviate a bit. 

        :raises: ValueError
        :rtype: int
        """
        cdef int i,size = self.size+1
        cdef double denom = 0.0
        cdef double[:] probs = self.probs
        cdef double before
        cdef double likelihood
        cdef double first_score

        ## find the denominator first
        for i in range(size):
            before = probs[i]

            ## keep track of 0th position in underlying baseline model 
            if i == 0: first_score = before
            elif i > 0 and before > first_score:
                self.first_rank += 1.0
            probs[i] = exp(before)
            denom += probs[i]

        ## normalize
        for i in range(size):
            probs[i] = probs[i]/denom

        ## likelihood (might want to make this more general)
        #likelihood = log(probs[0])
        if end > start:
            pass
        else:
            if start > size: likelihood = 0.0
            else:
                likelihood = log(probs[start])
        self.likelihood = likelihood

        ## check for overflow
        if isinf(likelihood) or isnan(likelihood):
            self.logger.fatal('Numerical overflow encountered!')
            raise ValueError('Numerical overflow encountered!')
        
## utility functions

def __report_features(path,hdescription,temp_sizes,starts,num):
    """Print information about features into working directory 

    :param path: working directory path 
    """
    ffile = os.path.join(path,"feature_info.txt")

    with open(ffile,'w') as finfo:
        for fn,description in hdescription.items():
            print >>finfo,"%d\t%s\t%d\tstarts at:%d" %\
              (fn,description,temp_sizes.get(fn,0),starts.get(fn,-1))
        print >>finfo,"\n\ntotal features: %d" % (num)

cdef void __print_features(str file_path, FeatureObj features,int[:] rvals,
                               int gold_id):
    """Prints the features to files, one line per 

    :param file_path: the path to print to
    """
    cdef int beam_size = len(features.features)
    cdef FeatureMap gold = features.gold_features
    cdef bint has_gold = True if gold else False
    cdef FeatureMap feat_map
    #cdef int n
    cdef long v

    with gzip.open(file_path+".gz",'wb') as feature_file:
        ## print the gold first
        if has_gold:
            print >>feature_file,"%d\t%s" %\
              (gold_id,' '.join(["%s=%f" % (n,v) for n,v in gold.iteritems()]))
        for i in range(beam_size):
            feat_map = features[i]
            print >>feature_file,"%d\t%s" %\
              (rvals[i],' '.join(["%s=%f" % (n,v) for n,v in feat_map.iteritems()]))
            
cdef FeatureObj __load_features(str file_path,str ftype,long maximum):
    """Loads a feature object from file using mapped memory

    :param file_path: the path to the file 
    :param ftype: type of feature set
    """

    cdef list contents,feature_maps = []
    cdef int k
    cdef dict templates = {}
    cdef FeatureMap fmap,gmap = FeatureMap(templates,maximum)
    cdef str entry
    cdef FeatureObj representation

    if ".gz" not in file_path:
        file_path += ".gz"

    with open(file_path) as handle:
        mapped = mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ)
        gzfile = gzip.GzipFile(mode='r',fileobj=mapped)
        contents = gzfile.readlines()

        for k,line in enumerate(contents):
            if k == 0 and 'train' in ftype:
                gmap = FeatureMap(templates,maximum)
                gmap.load_from_string(line.strip())
            else:
                fmap = FeatureMap(templates,maximum)
                fmap.load_from_string(line.strip())
                feature_maps.append(fmap)

    representation = FeatureObj(len(feature_maps),templates)
    representation.features = feature_maps
    representation.gold_features = gmap
    return representation
