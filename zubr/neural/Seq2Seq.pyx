# cython: profile=True

import re
import os
import shutil
import copy 
import time
import traceback
import logging
import random
import sys
cimport numpy as np
import numpy as np
from zubr.neural._dynet import save,load
from zubr.ZubrClass cimport ZubrSerializable
from zubr.neural.util import *
from zubr.util.config import ConfigAttrs
from zubr.SymmetricAlignment cimport SymmetricWordModel
from zubr.Alignment cimport SparseIBMM1
from cython cimport boundscheck,wraparound,cdivision

## dynet dependencies

from zubr.neural._dynet cimport (
    Expression,
    Trainer,
    ComputationGraph,
    ParameterCollection,
    LookupParameters,
    LSTMBuilder,
    Parameters,
    SimpleSGDTrainer,
    MomentumSGDTrainer,
    AdagradTrainer,
    AdamTrainer,
    RNNState,
    softmax,
    log,
    esum,
    tanh,
    concatenate,
    concatenate_cols,
    transpose,
    colwise_add,
    get_cg,
    select_rows,
    dropout,
    makeTensor,
    inputTensor,
    inputVector,
    logistic,
    cmult,
    sparsemax
)


cdef class Seq2SeqModel(ZubrSerializable):
    """Base class for Seq2Seq models.

    This is a pure cythonized version of: https://talbaumel.github.io/attention/
    """

    cdef Expression get_loss(self, int[:] x, int[:] z, ComputationGraph cg, double drop_out=0.0):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        :param computation graph 
        """
        raise NotImplementedError

    ##
    
    cdef list _embed_x(self,int[:] x,ComputationGraph cg):
        """Embed the given input for use in the neural model

        :param x: the input vector 
        :param cg: the computation graph
        """
        cdef LookupParameters embed = self.enc_embeddings
        cdef int i
        return [cg.lookup(embed,i,True) for i in x]

    cdef list _embed_z(self,int[:] z,ComputationGraph cg):
        """Embed the given input for use in the neural model

        :param z: the input vector 
        :param cg: the computation graph 
        """
        cdef LookupParameters embed = self.dec_embeddings
        cdef int i
        #return [cg.lookup(embed,i,True) for i in z if i != -1]
        return [cg.lookup(embed,i,True) for i in z]

    cdef list _run_enc_rnn(self,RNNState init_state,list input_vecs):
        """Run the encoder RNN with some initial state and input vector 

        :param init_state: the initial state 
        :param input_vecs: the input vectors
        """
        cdef RNNState s = init_state
        cdef list states,rnn_outputs

        ## cythonize this?
        states = s.add_inputs(input_vecs)
        rnn_outputs = [<Expression>st.output() for st in states]
        return rnn_outputs

    cdef Expression _get_probs(self,Expression rnn_output):
        """Get probabilities associated with RNN output
        
        :param rnn_output: the output of the rnn model 
        """
        cdef Expression output,bias,probs
        cdef Parameters output_w = self.output_w
        cdef Parameters output_b = self.output_b

        output = output_w.expr(True)
        bias = output_b.expr(True)
        probs = softmax(output*rnn_output+bias)
        return probs

    cdef EncoderInfo encode_input(self,int[:] x,ComputationGraph cg):
        """Compute a representation for a given input

        :param x: the input to encode 
        :param cg: the computation graph 
        """
        raise NotImplementedError

    def backup(self,wdir):
        """Back up the given model to file

        :param wdir: the working directory, or place to back up to
        :rtype: None 
        """
        raise NotImplementedError

    @classmethod
    def load_backup(cls,config,constraints=None):
        """Load a model instance from file 

        :param config: the main configuration 
        :returns: a seq2seq model instance 
        """
        raise NotImplementedError

    property hybrid:
        """Sets whether this is a hybrid neural model with some other type of model

        default is False
        """
        def __get__(self):
            return False
    

cdef class RNNSeq2Seq(Seq2SeqModel):
    pass

cdef class EncoderDecoder(RNNSeq2Seq):
    """Simple encoder-decoder model implementation"""

    def __init__(self,int enc_layers,
                     int dec_layers,
                     int embedding_size,
                     int enc_state_size,
                     int dec_state_size,
                     int enc_vocab_size,
                     int dec_vocab_size,
                     constraints=None,
                     interpolate=False,
                     lex_param=0.001,
                     interp_param=0.0,
                     ):
        """Create a simple encoder decoder instance 

        :param enc_layers: the number of layers used by the encoder RNN 
        :param dec_layers: the number of layers used by the decoder RNN
        :param embedding_size: the size of the embeddings used 
        :param enc_state_size: the size of the encoder RNN state size 
        :parma dec_state_size: the size of decoder RNN state size  
        """
        self.model = ParameterCollection()

        ## embedding parameters 
        self.enc_embeddings = self.model.add_lookup_parameters((enc_vocab_size,embedding_size))
        self.dec_embeddings = self.model.add_lookup_parameters((dec_vocab_size,embedding_size))

        ## RNN encode and decoder models 
        self.enc_rnn = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        self.dec_rnn = LSTMBuilder(dec_layers,enc_state_size,dec_state_size,self.model)

        ## output layer and bias for decoder RNN
        self.output_w = self.model.add_parameters((dec_vocab_size,dec_state_size))
        self.output_b = self.model.add_parameters((dec_vocab_size))

        ##lex model parameters
        self.interpolate = interpolate
        self.lex_param = lex_param
        self.interp_param = interp_param

        
    @classmethod
    def from_config(cls,config,constraints=None,lex=np.empty(0),copies={},lex2=np.empty(0)):
        """Create an encoder decoder instance from configuration 

        :param config: the global configuration 
        :param contraints: transition constraints (default == None)
        :rtype: EncodeDecoder 
        """
        stime = time.time()

        instance = cls(config.enc_rnn_layers,
                           config.dec_rnn_layers,
                           config.embedding_size,
                           config.enc_state_size,
                           config.dec_state_size,
                           config.enc_vocab_size,
                           config.dec_vocab_size,
                           constraints=constraints,
                           lex_model=lex,
                           lex_model2=lex2,
                           interpolate=config.interpolate,
                           lex_param=config.lex_param,
                           interp_param=config.interp_param,
                           copy=config.copy_mechanism,
                           copies=copies,
                           )

        instance.logger.info('Built model in %f seconds, embedding size=%d,enc vocab size=%d, dec vocab size=%d,trainer=%s,# copies %d' %\
                                 (time.time()-stime,
                                    config.embedding_size,
                                    config.enc_vocab_size,
                                    config.dec_vocab_size,
                                    config.trainer,
                                    len(copies),
                                 ))

        return instance

    ## c methods
    cdef RNNResult get_dec_distr_scratch(self,RNNState s,EncoderInfo e,Expression imatrix,int indx,ComputationGraph cg):
        """Get an output distribution from the decoder given input 

        :param s: the current RNN state 
        :param imatrix: the encoder side input matrix 
        :param idx: the index of the previous token 
        """
        raise NotImplementedError

    cdef RNNState append_state(self,RNNState s,EncoderInfo e, Expression imatrix,int identifier,ComputationGraph cg):
        """Create a new decoder state with an appended input

        :param s: the decoder state 
        :param e: information about the encoder 
        :param imatrix: matrix representation of encoder input 
        :param identifier: the identifier of the decoder token to add to state
        :param cg: the global computation graph
        """
        raise NotImplementedError

    #cdef list _encode_string(self,list embeddings,int[:] einput):
    cdef EncoderInfo _encode_string(self,list embeddings,int[:] einput):
        """Get the representationf for the input by running through RNN

        :param embeddings: the 
        """
        cdef LSTMBuilder enc_rnn = self.enc_rnn
        cdef RNNState initial_state = enc_rnn.initial_state()
        cdef list hidden_states

        ## copy stuff 
        cdef bint make_copy = self.copy
        cdef dict copies = self.copies,lookup = {}
        cdef int i,elen = einput.shape[0]

        ## annotations or hidden states
        hidden_states = self._run_enc_rnn(initial_state,embeddings)

        # mark the places where possible copies might be 
        if make_copy and copies:
            for i in range(elen):
                eid = einput[i]
                if eid in copies:
                    lookup[copies[eid]] = i

        return EncoderInfo(hidden_states,)

    cdef Expression get_loss(self, int[:] x, int[:] z, ComputationGraph cg, double drop_out=0.0):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef LSTMBuilder dec_rnn = self.dec_rnn
        cdef RNNState rnn_state
        cdef int w,zlen = z.shape[0]
        cdef Expression encoded,probs,loss_expr,total_loss
        cdef EncoderInfo e
        
        ## renew the computation graph directly 
        cg.renew(False,False,None)

        ## encode the input
        x_encoded = self._embed_x(x,cg)
        e = self._encode_string(x_encoded,x)[-1]
        #encoded = self._encode_string(x_encoded,x)[-1]
        encoded = e.encoded

        ##
        rnn_state = dec_rnn.initial_state()

        ## loop through the
        for w in range(zlen):
            if z[w] == -1: continue 
            rnn_state = rnn_state.add_input(encoded)
            probs = self._get_probs(rnn_state.output())
            loss_expr = -log(cg.outputPicker(probs,z[w],0))
            loss.append(loss_expr)
            
        total_loss = esum(loss)
        return total_loss

    cdef RNNState get_dec_init(self,ComputationGraph cg):
        """Get the initial state of the decoder RNN for generating output 

        :param cg: the computation graph 
        :returns: the initial RNN state 
        """
        cdef int enc_state_size = self.enc_state_size
        cdef LSTMBuilder dec_rnn = self.dec_rnn
        cdef LookupParameters embed = self.dec_embeddings
        cdef int dend = self._dend
        cdef Expression eos_embed
        cdef RNNState init 
        
        eos_embed = cg.lookup(embed,dend,True)
        init = dec_rnn.initial_state()
        return <RNNState>init.add_input(concatenate([cg.inputVector(enc_state_size*2),eos_embed]))

    cdef RNNResult get_dec_distr(self,RNNState s,EncoderInfo e,Expression imatrix,Expression last_embed):
        """Get an output distribution from the decoder 

        :param s: the current RNN state 
        :param imatrix: the encoder side input matrix 
        :param last_embed: the previous word embedding for decoder 
        """
        raise NotImplementedError

    cdef Expression get_init_embedding(self,ComputationGraph cg):
        """Get the initial state of the decoder RNN for generating output 

        :param cg: the computation graph 
        :returns: the initial RNN state 
        """
        cdef LookupParameters embed = self.dec_embeddings
        cdef int dend = self._dend
        cdef Expression eos_embed
        
        eos_embed = cg.lookup(embed,dend,True)
        return eos_embed 

    cdef Expression get_dec_embed(self,int idx,ComputationGraph cg):
        """Get the initial state of the decoder RNN for generating output 
        
        :param idx: the identifier 
        :param cg: the computation graph 
        """
        cdef LookupParameters embed = self.dec_embeddings
        cdef Expression embedding
        embedding = cg.lookup(embed,idx,True)
        return embedding

    property has_lex:
        """Determines if the model contains a lexical model """
        
        def __get__(self):
            """Returns true or false depending on whether alignment model is not empty

            :rtype: bint 
            """
            cdef np.ndarray lex = self.lex_model
            return lex.shape[0] > 0
            
cdef class AttentionModel(EncoderDecoder):

    def __init__(self,int enc_layers,
                     int dec_layers,
                     int embedding_size,
                     int enc_state_size,
                     int dec_state_size,
                     int enc_vocab_size,
                     int dec_vocab_size,
                     constraints=None,
                     np.ndarray lex_model=np.empty(0),
                     np.ndarray lex_model2=np.empty(0)
                     ):
        """Create a simple encoder decoder instance 

        :param enc_layers: the number of layers used by the encoder RNN 
        :param dec_layers: the number of layers used by the decoder RNN
        :param embedding_size: the size of the embeddings used 
        :param enc_state_size: the size of the encoder RNN state size 
        :parma dec_state_size: the size of decoder RNN state size  
        """
        EncoderDecoder.__init__(self,enc_layers,dec_layers,
                                    embedding_size,
                                    enc_state_size,dec_state_size,
                                    enc_vocab_size,dec_vocab_size)
        
        self.attention_w1 = self.model.add_parameters((enc_state_size,enc_state_size))
        self.attention_w2 = self.model.add_parameters((enc_state_size,dec_state_size))
        self.attention_v = self.model.add_parameters((1,enc_state_size))
        self.enc_state_size = enc_state_size
        self.lex_model = lex_model

    cdef Expression _attend(self,list input_vectors, RNNState state):
        """Runs the attention network to compute attentions cores

        :param input_vector 
        """
        cdef Parameters w1_o = self.attention_w1
        cdef Parameters w2_o = self.attention_w2
        cdef Parameters v_o = self.attention_v
        cdef Expression w1,w2,v,w2dt,normed
        cdef list weights = [],normalized = []
        cdef int input_vector,vlen = len(input_vectors)

        ## computations 
        cdef Expression attention_weight,new_v

        w1 = w1_o.expr(True)
        w2 = w2_o.expr(True)
        v = v_o.expr(True)
        w2dt = w2*((<tuple>state.h())[-1])

        for input_vector in range(vlen):
            attention_weight = v*tanh(w1*input_vectors[input_vector]+w2dt)
            weights.append(attention_weight)

        ## softmax normalization 
        normed = softmax(concatenate(weights))
        for input_vector in range(vlen):
            new_v = input_vectors[input_vector]*normed[input_vector]
            normalized.append(new_v)
        return esum(normalized)

    cdef Expression get_loss(self, int[:] x, int[:] z, ComputationGraph cg, double drop_out=0.0):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef LSTMBuilder dec_rnn = self.dec_rnn
        cdef RNNState rnn_state
        cdef int w,zlen = z.shape[0]
        cdef Expression probs,loss_expr,total_loss
        cdef int enc_state_size = self.enc_state_size
        cdef list encoded
        cdef EncoderInfo e
        
        ## renew the computation graph directly 
        cg.renew(False,False,None)

        x_encoded = self._embed_x(x,cg)
        e = self._encode_string(x_encoded,x)
        #encoded = self._encode_string(x_encoded,x)
        encoded = e.encoded
        
        rnn_state = dec_rnn.initial_state().add_input(cg.inputVector(enc_state_size))

        for w in range(zlen):
            
            ## skip over unknown words 
            if z[w] == -1: continue

            ## attention stuff 
            attended_encoding = self._attend(encoded,rnn_state)
            rnn_state = rnn_state.add_input(attended_encoding)
            probs = self._get_probs(rnn_state.output())
            loss_expr = -log(cg.outputPicker(probs,z[w],0))
            loss.append(loss_expr)

        total_loss = esum(loss)
        return total_loss

cdef class BiLSTMAttention(AttentionModel):
    """Attention model that uses a bidirectional LSTM model on the source side, 
    more in line with the original Bahdanau paper.

    Follows the attention.py example distributed in dynet/examples/python
    """

    def __init__(self,int enc_layers,
                     int dec_layers,
                     int embedding_size,
                     int enc_state_size,
                     int dec_state_size,
                     int enc_vocab_size,
                     int dec_vocab_size,
                     constraints=None,
                     np.ndarray lex_model=np.empty(0),
                     np.ndarray lex_model2=np.empty(0)                     
                     ):
        """Creates a BiLSTMAttention model instance 

        Note : this constructor doesn't inherit from the previous ones, 
        since it changes a few things, such as the how the decoder RNN 
        work. This should be generalized more...
        
        :param enc_layers: the number of encoder layers 
        :param dec_layers: the number of decoder layers 
        :param enc_state_size: the size of the encoder state 
        :param dec_state_size: the size of the decoder state size 
        :param enc_vocab_size: the size of the encoder vocabulary 
        :param dec_vocab_size: the size of the decoder vocabulary
        """
        self.model = ParameterCollection()

        ## embedding parameters 
        self.enc_embeddings = self.model.add_lookup_parameters((enc_vocab_size,embedding_size))
        self.dec_embeddings = self.model.add_lookup_parameters((dec_vocab_size,embedding_size))

        ## RNN encode and decoder models 
        self.enc_rnn     = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        self.dec_rnn     = LSTMBuilder(dec_layers,enc_state_size*2+embedding_size,dec_state_size,self.model)
        ## the reverse RNN
        self.enc_bwd_rnn = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        
        ## output layer and bias for decoder RNN
        self.output_w = self.model.add_parameters((dec_vocab_size,dec_state_size))
        self.output_b = self.model.add_parameters((dec_vocab_size))

        ## attention stuff
        self.attention_w1 = self.model.add_parameters((enc_state_size,enc_state_size*2))
        self.attention_w2 = self.model.add_parameters((enc_state_size,enc_state_size*enc_layers*2))
        self.attention_v = self.model.add_parameters((1,enc_state_size))
        self.enc_state_size = enc_state_size

        ### <EOS> positions
        self._eend = enc_vocab_size-1
        self._dend = dec_vocab_size-1

        ##
        self.lex_model = lex_model 

    cdef list _run_enc_rnn(self,RNNState init_state,list input_vecs):
        """Run the encoder RNN with some initial state and input vector 

        :param init_state: the initial state 
        :param input_vecs: the input vectors
        """
        cdef RNNState s = init_state
        cdef list states,rnn_outputs

        # ## cythonize this?
        states = s.add_inputs(input_vecs)
        rnn_outputs = [<Expression>st.output() for st in states]
        return rnn_outputs

    cdef EncoderInfo encode_input(self,int[:] x,ComputationGraph cg):
        """Compute a representation for a given input

        :param x: the input to encode 
        :param cg: the computation graph 
        """
        cdef list x_encoded = self._embed_x(x,cg)
        cdef EncoderInfo e
        e = self._encode_string(x_encoded,x)
        #encoded = self._encode_string(x_encoded,x)
        #encoded = e.encoded
        return e
        #return encoded

    cdef AttentionInfo _bi_attend(self,Expression input_matrix,RNNState s,Expression w1dt):
        """Playing around with the 

        :param input_matrix: the input "annotations" in matrix form 
        :param s: the decoder current state 
        :param w1dt: 
        """
        cdef Parameters w2_o = self.attention_w2
        cdef Parameters v_o = self.attention_v
        cdef Expression w2,v,w2dt,unormed,normed,context

        w2 = w2_o.expr(True)
        v = v_o.expr(True)
        w2dt = w2*concatenate(list(s.s()))
        unormed = transpose(v*tanh(colwise_add(w1dt,w2dt)))
        normed = softmax(unormed)
        context = input_matrix*normed
        
        #return context
        return AttentionInfo(normed,context,unormed)

    @boundscheck(False)
    cdef EncoderInfo _encode_string(self,list embeddings,int[:] einput):
        """Get the representationf for the input by running through RNN

        :param embeddings: the list of encoder embeddings 
        :param einput: the original encoder vector input 
        """
        cdef LSTMBuilder enc_rnn = self.enc_rnn
        cdef LSTMBuilder enc_bwd_rnn = self.enc_bwd_rnn
        cdef RNNState initial_state = enc_rnn.initial_state()
        cdef RNNState bwd_initial_state = enc_bwd_rnn.initial_state()
        cdef list fw_hidden_states,bwd_hidden_states
        cdef list vectors,sentence_rev = list(reversed(embeddings))
        cdef int i,j,vsize = len(embeddings)
        cdef Expression con
        cdef bint has_lex = self.has_lex
        cdef double[:,:] prob_lex
        cdef np.ndarray[ndim=2,dtype=np.double_t] lex
        cdef int dend = self._dend
        cdef int word_id

         ## copy stuff 
        cdef bint make_copy = self.copy
        cdef dict copies = self.copies,lookup = {}
        cdef int elen = einput.shape[0]

        ## hybrid
        cdef bint hybrid = self.hybrid

        ## run forward
        fw_hidden_states = self._run_enc_rnn(initial_state,embeddings)

        ## run backward and reverse
        bwd_hidden_states = self._run_enc_rnn(bwd_initial_state,sentence_rev)
        bwd_hidden_states = list(reversed(bwd_hidden_states))

        # mark the places where possible copies might be 
        if make_copy and copies:
            for i in range(elen):
                eid = einput[i]
                if eid in copies:
                    lookup[copies[eid]] = i

        ## create the final vectors
        vectors = []
        for i in range(vsize):
            con = concatenate([fw_hidden_states[i],bwd_hidden_states[i]])
            vectors.append(con)

        ## create lexical representation
        if has_lex or hybrid:
            prob_lex = self.lex_model

            ## might be slow 
            lex = np.zeros((dend+1,vsize),dtype='d')
            #with nogil:
            for i in range(dend-1):
                #for j in range(vsize): ## changed to remove scores for <EOS>
                for j in range(1,vsize-1):
                    word_id = 0 if einput[j] == -1 else einput[j]
                    lex[i][j] = prob_lex[word_id][i]
                    
            # ## return with
            return EncoderInfo(vectors,lex_probs=lex,lookup=lookup)                       
        # #return vectors
        return EncoderInfo(vectors,lookup=lookup)
        #return vectors

    cdef Expression get_loss(self, int[:] x, int[:] z, ComputationGraph cg, double drop_out=0.0):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef LSTMBuilder dec_rnn = self.dec_rnn
        cdef RNNState rnn_state
        cdef int i,w,zlen
        cdef Expression probs,loss_expr,total_loss
        cdef int enc_state_size = self.enc_state_size
        cdef EncoderInfo e        
        cdef list encoded #,actual_seq

        ## embedding on the target side 
        cdef Expression last_embed,input_mat

        ## parameterrs
        cdef Parameters w1_o = self.attention_w1
        cdef Parameters output_w = self.output_w
        cdef Parameters output_b = self.output_b
        cdef Expression w1,w1dt,vector,weight,b,out_vector

        ## renew the computation graph directly 
        cg.renew(False,False,None)

        ## parameter expressions
        w1 = w1_o.expr(True)
        weight = output_w.expr(True)
        b = output_b.expr(True)
        
        ## embed input x and run RNNs in both directions 
        x_encoded = self._embed_x(x,cg)
        #encoded = self._encode_string(x_encoded,x)
        e = self._encode_string(x_encoded,x)
        encoded = e.encoded
        ## create matrix of input vectors 
        input_mat = concatenate_cols(encoded)

        ## embed output z
        z_encoded = self._embed_z(z,cg)
        zlen = z.shape[0]
        last_embed = <Expression>z_encoded[0]

        ## rnn initial state
        rnn_state = dec_rnn.initial_state().add_input(concatenate([cg.inputVector(enc_state_size*2),last_embed]))

        ## start at 1
        for w in range(1,zlen):
            w1dt = w1*input_mat

            ## rep of [c_{i} ; z_{i-1}]
            #vector = dy.concatenate([self._bi_attend(input_mat, rnn_state, w1dt), last_embed])
            #vector = concatenate([self._bi_attend(input_mat, rnn_state, w1dt), last_embed])
            vector = concatenate([self._bi_attend(input_mat, rnn_state, w1dt).context, last_embed])
            ## LSTM()
            rnn_state = rnn_state.add_input(vector)
            out_vector = weight*rnn_state.output()+b

            ## MLP(c_{i},E(z-1),g_{i})
            
            probs = softmax(out_vector)
            #loss_expr = -log(cg.outputPicker(probs,actual_seq[w],0))
            loss_expr = -log(cg.outputPicker(probs,z[w],0))
            loss.append(loss_expr)
            last_embed = z_encoded[w]

        total_loss = esum(loss)
        return total_loss


cdef class AttentiveEncoderDecoder(BiLSTMAttention):

    """Attention model in the style of Luong et al (of my interpretation of it)"""

    def __init__(self,int enc_layers,
                     int dec_layers,
                     int embedding_size,
                     int enc_state_size,
                     int dec_state_size,
                     int enc_vocab_size,
                     int dec_vocab_size,
                     constraints=None,
                     np.ndarray lex_model=np.empty(0),
                     np.ndarray lex_model2=np.empty(0),
                     interpolate=False,
                     lex_param=0.001,
                     interp_param=0.0,
                     copy=False,
                     copies={},
                     ):
        """Creates a BiLSTMAttention model instance 

        Note : this constructor doesn't inherit from the previous ones, 
        since it changes a few things, such as the how the decoder RNN 
        work. This should be generalized more...
        

        :param enc_layers: the number of encoder layers 
        :param dec_layers: the number of decoder layers 
        :param enc_state_size: the size of the encoder state 
        :param dec_state_size: the size of the decoder state size 
        :param enc_vocab_size: the size of the encoder vocabulary 
        :param dec_vocab_size: the size of the decoder vocabulary
        """
        self.model = ParameterCollection()

        ## embedding parameters 
        self.enc_embeddings = self.model.add_lookup_parameters((enc_vocab_size,embedding_size))
        self.dec_embeddings = self.model.add_lookup_parameters((dec_vocab_size,embedding_size))

        ## RNN encode and decoder models 
        self.enc_rnn     = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        self.dec_rnn     = LSTMBuilder(dec_layers,enc_state_size*2+embedding_size,dec_state_size,self.model)
        ## the reverse RNN
        self.enc_bwd_rnn = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        
        ## output layer and bias for decoder RNN
        self.output_w = self.model.add_parameters((dec_vocab_size,dec_state_size+(enc_state_size*2)))
        self.output_b = self.model.add_parameters((dec_vocab_size))

        ## one more mlp
        self.output_final = self.model.add_parameters((dec_vocab_size,dec_vocab_size))
        self.final_bias = self.model.add_parameters((dec_vocab_size))

        ## attention stuff
        self.attention_w1 = self.model.add_parameters((enc_state_size,enc_state_size*2))
        self.attention_w2 = self.model.add_parameters((enc_state_size,enc_state_size*enc_layers*2))
        self.attention_v = self.model.add_parameters((1,enc_state_size))
        self.enc_state_size = enc_state_size

        ### <EOS> positions
        self._eend = enc_vocab_size-1
        self._dend = dec_vocab_size-1
        
        ##
        self.lex_model = lex_model
        self.interpolate = interpolate
        self.lex_param = lex_param
        self.interp_param = interp_param
        has_lex = lex_model.shape[0] > 0

        ## copy parameter
        self.copy = copy
        self.copies = copies

        ## log information for debugging purposes 
        self.logger.info('Initialized model, has_lex=%s,interpolate=%s, lex_param=%f, interp_param=%f,copy=%s' %\
                             (has_lex,str(interpolate),lex_param,interp_param,str(copy)))
 
    cdef Expression get_loss(self, int[:] x, int[:] z, ComputationGraph cg, double drop_out=0.0):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef RNNState rnn_state
        cdef int w,zlen
        cdef Expression probs,loss_expr,total_loss
        cdef list encoded
        cdef Expression last_embed,input_mat,output_layer
        cdef RNNResult result

        ## encoder information 
        cdef EncoderInfo e
        cdef dict lookup
        
        ## copy stuff
        cdef bint compute_copy = self.copy
        cdef Expression target_val
        cdef double tval,amax
        ## attention values
        cdef Expression attention_weights,new_softmax
        cdef double[:] new_vals
        cdef int val_len
        
        ## renew the computation graph directly 
        cg.renew(False,False,None)

        ## embed input x and run RNNs in both directions 
        x_encoded = self._embed_x(x,cg)
        
        #encoded = self._encode_string(x_encoded,x)
        e = self._encode_string(x_encoded,x)
        lookup = e.lookup
        encoded = e.encoded

        ## create matrix of input vectors 
        input_mat = concatenate_cols(encoded)

        ## embed output z
        z_encoded = self._embed_z(z,cg)
        zlen = z.shape[0]

        ## rnn initial state
        last_embed = z_encoded[0]
        rnn_state = self.get_dec_init(cg)

        ## note : ignores unknown words
        for w in range(1,zlen):

            ## get the disitrubution over 
            result = self.get_dec_distr(rnn_state,e,input_mat,last_embed)
            output_layer = result.probs

            ## drop out 
            if drop_out > 0.0:
                output_layer = dropout(output_layer,drop_out)

            if compute_copy:

                ## compute a different softmax
                attention_weights = result.attention_scores
                new_softmax = softmax(<Expression>concatenate([output_layer,attention_weights]))
                #new_vals = new_softmax.npvalue()
                new_vals = output_layer.npvalue()
                val_len = new_vals.shape[0]

                ## a potential copy here 
                if z[w] in lookup:

                    target_val = cg.outputPicker(new_softmax,z[w],0)
                    npicked = cg.outputPicker(new_softmax,val_len+lookup[z[w]],0)
                    amax = npicked.scalar_value()
                    tval = target_val.scalar_value()

                    ## if copy score is larger, use it
                    if tval <= amax: target_val = npicked

                ## no copy in this case  
                else:
                    target_val = cg.outputPicker(new_softmax,z[w],0)
                    
            else:
                probs = softmax(output_layer)
                target_val = cg.outputPicker(probs,z[w],0)

            loss_expr = -log(target_val)
            loss.append(loss_expr)
            rnn_state = result.state
            last_embed = z_encoded[w]

        total_loss = esum(loss)
        return total_loss

    cdef RNNResult get_dec_distr(self,RNNState s,
                                     EncoderInfo e,
                                     Expression imatrix,
                                     Expression last_embed):
        """Get an output distribution from the decoder given input 

        :param s: the current RNN state 
        :param imatrix: the encoder side input matrix 
        :param last_embed: the previous word embedding for decoder 
        """
        cdef Parameters w1_o       = self.attention_w1
        cdef Parameters output_w   = self.output_w
        cdef Parameters output_b   = self.output_b
        cdef Parameters output_f   = self.output_final
        cdef Parameters final_bias = self.final_bias
        cdef Expression w1,w1dt,vector,weight,b,out_vector,context_vector,joined
        cdef Expression inside_mpe
        cdef Expression ofinal,obias,input_mat,probs
        cdef RNNState rnn_state
        cdef AttentionInfo attention

        ## lexical information 
        cdef bint has_lex = e.has_lex
        cdef Expression aweights,lex_probs,p_l
        cdef double lex_param = self.lex_param

        ## interpolate 
        cdef bint interpolate = self.interpolate
        cdef Expression p_l_sm,p_m_sm
        cdef double it,interp_param = self.interp_param

        ## copy
        cdef bint make_copy = self.copy
        cdef bint hybrid = self.hybrid

        ## gate expressions if exist
        cdef Parameters gate,gate_bias
        
        ## parameter expressions
        w1     = w1_o.expr(True)
        weight = output_w.expr(True)
        b      = output_b.expr(True)
        ofinal = output_f.expr(True)
        obias  = final_bias.expr(True)
        w1dt   = w1*imatrix

        ## context vector c_{i} = attention MLP
        attention = self._bi_attend(imatrix,s,w1dt)
        context_vector = attention.context_vector
        vector = concatenate([context_vector, last_embed])

        ## g_{i} = LSTM(g_{i-1},[ c_{i} ; E_{z_{i-1}} ])
        rnn_state = s.add_input(vector)
        out_vector = rnn_state.output()

        ## o = W * MLP(g_{i}, c_{i}) + b
        inside_mpe = tanh(weight*concatenate([out_vector,context_vector])+b)
        
        ## the attention vector 
        aweights = attention.attention_weights

        if has_lex and not self.hybrid:
            
            lex_probs = e.lex_probs

            ## interpolate the scores (not recommended)
            if interpolate:

                ## the iterpolation parameter 
                it = sigmoid(interp_param)
                p_m_sm = ofinal*inside_mpe+obias
                p_l_sm = lex_probs*aweights
                joined = concatenate_cols([p_l_sm,p_m_sm])*inputVector(np.array([[it],[1.0-it]]))

            ## use the bias 
            else:
                p_l = (lex_probs*aweights)+lex_param
                joined = ofinal*inside_mpe+obias+log(p_l)
        else: 
            joined = ofinal*inside_mpe+obias

        return RNNResult(joined,rnn_state,attention.unormalized,make_copy)

    cdef RNNState append_state(self,RNNState s,EncoderInfo e, Expression imatrix,int identifier,ComputationGraph cg):
        """Create a new decoder state with an appended input

        :param s: the decoder state 
        :param e: information about the encoder 
        :param imatrix: matrix representation of encoder input 
        :param identifier: the identifier of the decoder token to add to state
        :param cg: the global computation graph
        """
        cdef Expression last_embed = self.get_dec_embed(identifier,cg)
        cdef RNNState rnn_state
        cdef AttentionInfo attention
        cdef Expression w1dt,w1
        cdef Parameters w1_o = self.attention_w1

        ## attention stuff 
        w1     = w1_o.expr(True)
        w1dt   = w1*imatrix

        ## context vector c_{i} = attention MLP
        attention = self._bi_attend(imatrix,s,w1dt)
        context_vector = attention.context_vector
        vector = concatenate([context_vector, last_embed])

        ## the new state 
        rnn_state = s.add_input(vector)
        return rnn_state

    cdef RNNResult get_dec_distr_scratch(self,RNNState s,EncoderInfo e,Expression imatrix,int indx,ComputationGraph cg):
        """Get an output distribution from the decoder given input 

        :param s: the current RNN state 
        :param imatrix: the encoder side input matrix 
        :param idx: the index of the previous token 
        """
        cdef Expression last_embed = self.get_dec_embed(indx,cg)
        return <RNNResult>self.get_dec_distr(s,e,imatrix,last_embed)
    
    def backup(self,wdir):
        """Back up the given model to file

        :param wdir: the working directory, or place to back up to
        :rtype: None 
        """
        model_out = os.path.join(wdir,"attention_model")
        stime = time.time()
        
        save(model_out,[
            self.enc_embeddings,
            self.dec_embeddings,
            self.output_w,
            self.output_b,
            self.enc_rnn,
            self.enc_bwd_rnn,
            self.dec_rnn,
            self.attention_w1,
            self.attention_w2,
            self.attention_v,
            self.output_final,
            self.final_bias
        ])

        ## backup the lex model
        lex_out = os.path.join(wdir,"lex_model")
        np.savez_compressed(lex_out,self.lex_model,self.copies)
        self.logger.info('Finished saving in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config,constraints=None):
        """Load a model instance from file 

        :param config: the main configuration 
        :returns: a seq2seq model instance 
        """
        cdef AttentiveEncoderDecoder instance
        
        loc = os.path.join(config.dir,"attention_model")
        stime = time.time()

        ## lex model
        lex_out = os.path.join(config.dir,"lex_model.npz")
        archive = np.load(lex_out)
        lex_model = archive["arr_0"]

        ## copies
        copies = archive["arr_1"].item()
        if copies: copy_mechanism = True
        else: copy_mechanism = False


        ## create empty instance
        instance = cls(config.enc_rnn_layers,
                           config.dec_rnn_layers,
                           config.embedding_size,
                           config.enc_state_size,
                           config.dec_state_size,
                           config.enc_vocab_size,
                           config.dec_vocab_size,
                           constraints=constraints,
                           lex_model=lex_model, ## need to add additional stuff here
                           copy=copy_mechanism,
                           copies=copies,
                           lex_param=config.lex_param
                           )

        
        ## load all of the components 
        enc_e,dec_e,output_w,output_b,enc_rnn,enc_bwd_rnn,dec_rnn,a1,a2,av,f,fb = load(loc,instance.model)

        # reload all the stuff
        instance.enc_embeddings = enc_e
        instance.dec_embeddings = dec_e
        instance.enc_rnn = enc_rnn
        instance.enc_bwd_rnn = enc_bwd_rnn
        instance.dec_rnn = dec_rnn
        instance.output_w = output_w
        instance.output_b = output_b
        instance.output_final = f
        instance.final_bias = fb
        instance.attention_w1 = a1
        instance.attention_w2 = a2
        instance.attention_v = av

        ## log the load time 
        #stance.logger.info('Loaded model in %s seconds' % (str(time.time()-stime)))

        instance.logger.info('Built model in %f seconds, embedding size=%d,enc vocab size=%d, dec vocab size=%d,trainer=%s' %\
                                 (time.time()-stime,
                                    config.embedding_size,
                                    config.enc_vocab_size,
                                    config.dec_vocab_size,
                                    config.trainer))
        
        return instance


cdef class ConstrainedAttention(AttentiveEncoderDecoder):
    """A model where the output vocabulary distributions on the decoder side 
    are constrained to known transitions from the training data, which hopefully 
    will make the trainer much faster by requiring smaller vector to run softmaxes 
    over 
    """


    def __init__(self,int enc_layers,
                     int dec_layers,
                     int embedding_size,
                     int enc_state_size,
                     int dec_state_size,
                     int enc_vocab_size,
                     int dec_vocab_size,
                     constraints=None,
                     np.ndarray lex_model=np.empty(0),
                     np.ndarray lex_model2=np.empty(0),
                     interpolate=False,
                     lex_param=0.001,
                     interp_param=0.0,
                     copy=False,
                     copies={},
                     ):
        """Creates a BiLSTMAttention model instance 

        Note : this constructor doesn't inherit from the previous ones, 
        since it changes a few things, such as the how the decoder RNN 
        work. This should be generalized more...
        
        :param enc_layers: the number of encoder layers 
        :param dec_layers: the number of decoder layers 
        :param enc_state_size: the size of the encoder state 
        :param dec_state_size: the size of the decoder state size 
        :param enc_vocab_size: the size of the encoder vocabulary 
        :param dec_vocab_size: the size of the decoder vocabulary
        """
        self.model = ParameterCollection()

        ## embedding parameters 
        self.enc_embeddings = self.model.add_lookup_parameters((enc_vocab_size,embedding_size))
        self.dec_embeddings = self.model.add_lookup_parameters((dec_vocab_size,embedding_size))

        ## RNN encode and decoder models 
        self.enc_rnn     = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        self.dec_rnn     = LSTMBuilder(dec_layers,enc_state_size*2+embedding_size,dec_state_size,self.model)
        ## the reverse RNN
        self.enc_bwd_rnn = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)

        ## output layer and bias for decoder RNN
        self.output_w = self.model.add_parameters((dec_vocab_size,dec_state_size+(enc_state_size*2)))
        self.output_b = self.model.add_parameters((dec_vocab_size))


        ## one more mlp
        self.output_final = self.model.add_parameters((dec_vocab_size,dec_vocab_size))
        self.final_bias = self.model.add_parameters((dec_vocab_size))


        ## attention stuff
        self.attention_w1 = self.model.add_parameters((enc_state_size,enc_state_size*2))
        self.attention_w2 = self.model.add_parameters((enc_state_size,enc_state_size*enc_layers*2))
        self.attention_v = self.model.add_parameters((1,enc_state_size))
        self.enc_state_size = enc_state_size


        ### <EOS> positions
        self._eend = enc_vocab_size-1
        self._dend = dec_vocab_size-1

        ## transitions
        self.transitions = constraints

        ##
        self.lex_model = lex_model
        self.interpolate = interpolate
        self.lex_param = lex_param
        self.interp_param = interp_param
        has_lex = lex_model.shape[0] > 0

        ## copying parameter
        self.copy = copy
        self.copies = copies 

        ## log information for debugging purposes 
        self.logger.info('Initialized model, has_lex=%s,interpolate=%s, lex_param=%f, interp_param=%f,copy=%s, #copies=%d' %\
                             (has_lex,str(interpolate),lex_param,interp_param,str(copy),len(copies)))

    cdef Expression get_loss(self, int[:] x, int[:] z, ComputationGraph cg, double drop_out=0.0):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef RNNState rnn_state
        cdef int w,zlen
        cdef Expression selected,probs,output_layer,loss_expr,total_loss
        cdef list encoded
        cdef Expression last_embed,input_mat
        cdef RNNResult result
        cdef TransitionTable transitions = self.transitions
        cdef TransitionPair constraints
        cdef np.ndarray tlist
        cdef dict tlookup

        ### word information
        cdef int prev_word
        cdef EncoderInfo e

        ## copy stuff
        cdef bint compute_copy = self.copy
        cdef Expression target_val,attention_weights
        cdef double tval,amax
        cdef Expression new_softmax,npicked
        
        ## renew the computation graph directly 
        cg.renew(False,False,None)

        ## embed input x and run RNNs in both directions 
        x_encoded = self._embed_x(x,cg)
        e = self._encode_string(x_encoded,x)
        lookup = e.lookup 
        encoded = e.encoded
        
        ## create matrix of input vectors 
        input_mat = concatenate_cols(encoded)

        ## embed output z
        z_encoded = self._embed_z(z,cg)
        zlen = z.shape[0]

        ## rnn initial state
        last_embed = z_encoded[0]
        prev_word = z[0] 
        rnn_state = self.get_dec_init(cg)

        ## note : ignores unknown words
        for w in range(1,zlen):

            ## compute distribution over output vocab
            result = self.get_dec_distr(rnn_state,e,input_mat,last_embed)
            output_layer = result.probs

            ## get the constraint on allowable output 
            constraints = transitions.get_constraints(prev_word)
            tlist = constraints.tlist
            tlookup = constraints.tlookup

            ## a selection of the possibly outputs 
            selected = <Expression>select_rows(output_layer,tlist)

            if compute_copy:

                ## new softmax computation 
                attention_weights = result.attention_scores
                new_softmax = softmax(<Expression>concatenate([selected,attention_weights]))

                # # ## has a potential copy 
                if z[w] in lookup:

                    target_val = cg.outputPicker(new_softmax,tlookup[z[w]],0)
                    npicked = cg.outputPicker(new_softmax,len(tlist)+lookup[z[w]],0)
                    amax = npicked.scalar_value()
                    tval = target_val.scalar_value()

                    ## if copy score is larger, use it
                    if tval <= amax: target_val = npicked

                ## no copy in this case  
                else:
                    target_val = cg.outputPicker(new_softmax,tlookup[z[w]],0)

            else:
                probs = softmax(selected)
                target_val = cg.outputPicker(probs,tlookup[z[w]],0)

            loss_expr = -log(target_val)
            loss.append(loss_expr)
            last_embed = z_encoded[w]

            rnn_state = result.state
            prev_word = z[w] if z[w] != -1 else 0

        total_loss = esum(loss)
        return total_loss


cdef class AttentionLexModel(ConstrainedAttention):
    """

    This model implements a joint lexical translation and neural model 
    that works by interpolating the results from both models to make predictions. 
    The parameters associated with the interpolation are learned using some extra
    neural machinery

    This is based on the model from: 
    
    Wang et al. Neural Machine Translation Advised by Statistical Machine Translation, 
    AAAI 2017

    In this approach, generating a next word distribution on the decoder side is computed 
    in the following way: 

    p( z_{i} | z_{< i}, x) = (1 - \alpha) p_nmt(z_{i} | z_{< i},x) + \alpha p_lex( z_{i} | z_{< i}, x)

    where the alpha is computed using an additional non-linear function or network: 

    a = sigmoid(f(s_{i},z_{i-1},c_{i})) 

    My interpretation of this network is that it learns the reliability of the neural model 
    at the particular given moment when a prediction is needed. It's not entirely clear, however, 
    why the first model is interpolated with 1 - alpha as opposed to just alpha 


    """
    def __init__(self,int enc_layers,
                     int dec_layers,
                     int embedding_size,
                     int enc_state_size,
                     int dec_state_size,
                     int enc_vocab_size,
                     int dec_vocab_size,
                     constraints=None,
                     np.ndarray lex_model=np.empty(0),
                     np.ndarray lex_model2=np.empty(0),
                     interpolate=False,
                     lex_param=0.001,
                     interp_param=0.0,
                     copy=False,
                     copies={},
                     ):
        """Creates a BiLSTMAttention model instance 

        Note : this constructor doesn't inherit from the previous ones, 
        since it changes a few things, such as the how the decoder RNN 
        work. This should be generalized more...
        
        :param enc_layers: the number of encoder layers 
        :param dec_layers: the number of decoder layers 
        :param enc_state_size: the size of the encoder state 
        :param dec_state_size: the size of the decoder state size 
        :param enc_vocab_size: the size of the encoder vocabulary 
        :param dec_vocab_size: the size of the decoder vocabulary
        """
        self.model = ParameterCollection()

        ## embedding parameters 
        self.enc_embeddings = self.model.add_lookup_parameters((enc_vocab_size,embedding_size))
        self.dec_embeddings = self.model.add_lookup_parameters((dec_vocab_size,embedding_size))

        ## RNN encode and decoder models 
        self.enc_rnn     = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        self.dec_rnn     = LSTMBuilder(dec_layers,enc_state_size*2+embedding_size,dec_state_size,self.model)
        ## the reverse RNN
        self.enc_bwd_rnn = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        
        ## output layer and bias for decoder RNN
        self.output_w = self.model.add_parameters((dec_vocab_size,dec_state_size+(enc_state_size*2)))
        self.output_b = self.model.add_parameters((dec_vocab_size))

        ## one more mlp
        self.output_final = self.model.add_parameters((dec_vocab_size,dec_vocab_size))
        self.final_bias = self.model.add_parameters((dec_vocab_size))

        ## attention stuff
        self.attention_w1 = self.model.add_parameters((enc_state_size,enc_state_size*2))
        self.attention_w2 = self.model.add_parameters((enc_state_size,enc_state_size*enc_layers*2))
        self.attention_v = self.model.add_parameters((1,enc_state_size))
        self.enc_state_size = enc_state_size

        ### <EOS> positions
        self._eend = enc_vocab_size-1
        self._dend = dec_vocab_size-1

        ## transitions
        self.transitions = constraints

        ## bidirectional lexical models 
        self.lex_model = lex_model
        self.lex_model2 = lex_model2
        
        self.interpolate = interpolate
        self.lex_param = lex_param
        self.interp_param = interp_param
        has_lex  = lex_model.shape[0]  > 0
        has_lex2 = lex_model2.shape[0] > 0

        ## copying parameter
        self.copy = copy
        self.copies = copies

        ## gated network
        # self.gate = self.model.add_parameters((dec_vocab_size,dec_state_size+(enc_state_size*2)+embedding_size))
        # self.gate_bias = self.model.add_parameters((dec_vocab_size))
        self.gate = self.model.add_parameters((1,dec_state_size+(enc_state_size*2)+embedding_size))
        self.gate_bias = self.model.add_parameters((1))

        ## log information for debugging purposes 
        self.logger.info('Initialized model, has_lex=%s, has_lex2=%s, interpolate=%s, lex_param=%f, interp_param=%f' %\
                             (has_lex,has_lex2,str(interpolate),lex_param,interp_param))
       

    ## backup protocol

    def backup(self,wdir):
        """Back up the given model to file

        :param wdir: the working directory, or place to back up to
        :rtype: None 
        """
        model_out = os.path.join(wdir,"attention_model")
        stime = time.time()
        
        save(model_out,[
            self.enc_embeddings,
            self.dec_embeddings,
            self.output_w,
            self.output_b,
            self.enc_rnn,
            self.enc_bwd_rnn,
            self.dec_rnn,
            self.attention_w1,
            self.attention_w2,
            self.attention_v,
            self.output_final,
            self.final_bias
        ])

        ## backup the lex model
        lex_out = os.path.join(wdir,"lex_model")
        np.savez_compressed(lex_out,self.lex_model,self.copies,self.lex_model2)
        self.logger.info('Finished saving in %s seconds' % str(time.time()-stime))

    ### new loss function

    cdef RNNResult get_dec_distr(self,RNNState s,
                                     EncoderInfo e,
                                     Expression imatrix,
                                     Expression last_embed):
        """Get an output distribution from the decoder given input 

        :param s: the current RNN state 
        :param imatrix: the encoder side input matrix 
        :param last_embed: the previous word embedding for decoder 
        """
        cdef Parameters w1_o       = self.attention_w1
        cdef Parameters output_w   = self.output_w
        cdef Parameters output_b   = self.output_b
        cdef Parameters output_f   = self.output_final
        cdef Parameters final_bias = self.final_bias
        cdef Expression w1,w1dt,vector,weight,b,out_vector,context_vector,joined
        cdef Expression inside_mpe
        cdef Expression ofinal,obias,input_mat,probs
        cdef RNNState rnn_state
        cdef AttentionInfo attention

        ## lexical information 
        cdef bint has_lex = e.has_lex
        cdef Expression aweights,lex_probs,p_l
        cdef double lex_param = self.lex_param

        ## interpolate 
        cdef Expression p_l_sm,p_m_sm
        cdef double it,interp_param = self.interp_param

        ## copy
        cdef bint make_copy = self.copy
        cdef bint hybrid = self.hybrid

        ## gate expressions if exist
        cdef Parameters gate = self.gate
        cdef Parameters gate_bias = self.gate_bias
        cdef Expression gate_weight,gate_last,gate_info
        
        ## parameter expressions
        w1     = w1_o.expr(True)
        weight = output_w.expr(True)
        b      = output_b.expr(True)
        ofinal = output_f.expr(True)
        obias  = final_bias.expr(True)
        w1dt   = w1*imatrix

        ## gated parameters
        gate_weight = gate.expr(True)
        gate_last = gate_bias.expr(True)
        
        ## context vector c_{i} = attention MLP
        attention = self._bi_attend(imatrix,s,w1dt)
        context_vector = attention.context_vector
        vector = concatenate([context_vector, last_embed])

        ## g_{i} = LSTM(g_{i-1},[ c_{i} ; E_{z_{i-1}} ])
        rnn_state = s.add_input(vector)
        out_vector = rnn_state.output()

        ## o = W * MLP(g_{i}, c_{i}) + b
        inside_mpe = tanh(weight*concatenate([out_vector,context_vector])+b)
        
        ## the attention vector 
        aweights = attention.attention_weights
        
        if has_lex:
            lex_probs = e.lex_probs
            p_l = (lex_probs*aweights)+lex_param
            joined = ofinal*inside_mpe+obias+log(p_l)
        else: 
            joined = ofinal*inside_mpe+obias

        ## gate mechanism
        gate_info = tanh(gate_weight*concatenate([out_vector,last_embed,context_vector])+gate_last)

        return <RNNResult>HybridRNNResult(joined,rnn_state,attention.unormalized,make_copy,gate_info)

    cdef Expression get_loss(self, int[:] x, int[:] z, ComputationGraph cg, double drop_out=0.0):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef RNNState rnn_state
        cdef int w,zlen
        cdef Expression selected,probs,neural_probs,output_layer,loss_expr,total_loss
        cdef list encoded
        cdef Expression last_embed,input_mat
        #cdef RNNResult result
        cdef HybridRNNResult result
        cdef TransitionTable transitions = self.transitions
        cdef TransitionPair constraints
        cdef np.ndarray tlist
        cdef dict tlookup

        ### word information
        cdef int prev_word
        cdef EncoderInfo e

        ## copy stuff
        cdef bint compute_copy = self.copy
        cdef Expression target_val,attention_weights
        cdef double tval,amax
        cdef Expression new_softmax,npicked

        ## gate stuff
        cdef Expression gate,lex_distribution
        
        ## input output len
        cdef int xlen = x.shape[0],i,j
        cdef int tlen,node_name
        cdef double rscore,tscore,tsum,selected_gate

        ## lexical information
        cdef double[:,:] lex_scores
        cdef np.ndarray[ndim=1,dtype=np.double_t] lex_vote
        cdef int lex_size
        cdef double tdenom
        
        
        ## renew the computation graph directly 
        cg.renew(False,False,None)

        ## embed input x and run RNNs in both directions 
        x_encoded = self._embed_x(x,cg)
        e = self._encode_string(x_encoded,x)
        lookup = e.lookup 
        encoded = e.encoded
        
        ## create matrix of input vectors 
        input_mat = concatenate_cols(encoded)

        ## embed output z
        z_encoded = self._embed_z(z,cg)
        zlen = z.shape[0]

        ## rnn initial state
        last_embed = z_encoded[0]
        prev_word = z[0] 
        rnn_state = self.get_dec_init(cg)

        ## note : ignores unknown words
        for w in range(1,zlen):

            ## compute distribution over output vocab
            result = self.get_dec_distr(rnn_state,e,input_mat,last_embed)
            output_layer = result.probs

            ## get the constraint on allowable output 
            constraints = transitions.get_constraints(prev_word)
            tlist = constraints.tlist
            tlookup = constraints.tlookup
            tlen = len(tlist) 

            ## a selection of the possibly outputs 
            selected = <Expression>select_rows(output_layer,tlist)

            ## neural model distribution 
            neural_probs = softmax(selected)

            ## run the gate
            gate = result.gate
            #selected_gate = select_rows(gate,tlist)
            selected_gate = sigmoid(gate.npvalue()[0])

            ##
            lex_scores = (select_rows(<Expression>e.lex_probs,tlist)).npvalue()
            lex_size = lex_scores.shape[0]
            lex_vote = np.zeros((tlen,),dtype='d')

            ## take the max score
            tdenom = 0.0
            
            for i in range(lex_size):
                #lex_vote[i] = np.max(lex_scores[i])
                lex_vote[i] = np.sum(lex_scores[i])
                tdenom += lex_vote[i]

            ## normalize
            for i in range(lex_size):
                if tdenom == 0.0: lex_vote[i] = 1.0/float(tlen)
                elif tdenom != 0.0: lex_vote[i] = lex_vote[i]/tdenom

            ## lex distribution
            lex_distribution = inputVector(lex_vote)

            probs = concatenate_cols([neural_probs,lex_distribution])*\
              inputVector(np.array([[0.5],[0.5]]))

            ##
            #target_val = cg.outputPicker(neural_probs,tlookup[z[w]],0)
            target_val = cg.outputPicker(probs,tlookup[z[w]],0)

            loss_expr = -log(target_val)
            loss.append(loss_expr)
            last_embed = z_encoded[w]
            rnn_state = result.state
            prev_word = z[w] if z[w] != -1 else 0

        total_loss = esum(loss)
        return total_loss
    

    @classmethod
    def load_backup(cls,config,constraints=None):
        """Load a model instance from file 

        :param config: the main configuration 
        :returns: a seq2seq model instance 
        """
        cdef AttentiveEncoderDecoder instance
        
        loc = os.path.join(config.dir,"attention_model")
        stime = time.time()

        ## lex model
        lex_out = os.path.join(config.dir,"lex_model.npz")
        archive = np.load(lex_out)
        lex_model = archive["arr_0"]
        ## lex model 2
        lex_model2 = archive["arr_2"]

        ## copies
        copies = archive["arr_1"].item()
        if copies: copy_mechanism = True
        else: copy_mechanism = False

        ## create empty instance
        instance = cls(config.enc_rnn_layers,
                           config.dec_rnn_layers,
                           config.embedding_size,
                           config.enc_state_size,
                           config.dec_state_size,
                           config.enc_vocab_size,
                           config.dec_vocab_size,
                           constraints=constraints,
                           lex_model=lex_model,
                           lex_model2=lex_model2,
                           copy=copy_mechanism,
                           copies=copies,
                           lex_param=config.lex_param
                           )

        ## load all of the components 
        enc_e,dec_e,output_w,output_b,enc_rnn,enc_bwd_rnn,dec_rnn,a1,a2,av,f,fb = load(loc,instance.model)

        # reload all the stuff
        instance.enc_embeddings = enc_e
        instance.dec_embeddings = dec_e
        instance.enc_rnn = enc_rnn
        instance.enc_bwd_rnn = enc_bwd_rnn
        instance.dec_rnn = dec_rnn
        instance.output_w = output_w
        instance.output_b = output_b
        instance.output_final = f
        instance.final_bias = fb
        instance.attention_w1 = a1
        instance.attention_w2 = a2
        instance.attention_v = av

        ## log the load time 
        #stance.logger.info('Loaded model in %s seconds' % (str(time.time()-stime)))

        instance.logger.info('Built model in %f seconds, embedding size=%d,enc vocab size=%d, dec vocab size=%d,trainer=%s' %\
                                 (time.time()-stime,
                                    config.embedding_size,
                                    config.enc_vocab_size,
                                    config.dec_vocab_size,
                                    config.trainer))
        
        return instance


    property hybrid:
        """Sets whether this is a hybrid neural model with some other type of model

        default is False
        """
        def __get__(self):
            return True

## seq to seq learners



cdef class Seq2SeqLearner(ZubrSerializable):
    """Class for training Seq2Seq models"""

    def __init__(self,trainer,model,stable,config):
        """Creates a seq2seq learner

        :param model: the underlying neural model 
        :param train_data: the training data 
        :param valid_data: the validation data
        :param stable: the symbol table 
        :param eend: the encoder end symbol 
        :param dend: the decoder end symbol 
        """
        self.trainer = <Trainer>trainer 
        self.model   = <Seq2SeqModel>model
        self.stable  = stable
        self.cg      = get_cg()
        self._config = config
        
    def log_epoch(self,inum,itime,tloss,vtime,vloss,tsize,vsize):
        """Log information related to the epoch, including -log likelihood, perplexity, epoch time, etc..

        :param inum: iteration number
        :param itime: the iteration time 
        :param tloss: the training loss 
        :param vtime: validation time (if available)
        :param vloss: validation lost (if available) 
        :param tsize: number of training target words in data 
        :param vsize: number of target words in validation (if available)
        :rtype: None
        """
        tppl = np.exp(tloss/tsize)
        vppl = np.exp(vloss/vsize)
        
        self.logger.info('Finished iteration %d in %s seconds, ran val in %s seconds, train_loss=%s, train ppl=%s, val loss=%s, val ppl=%s' % (inum,str(itime),str(vtime),str(tloss),str(tppl),str(vloss),str(vppl)))

    ## training methods
    cpdef void train(self,config):
        """Trian the model using data 

        :param config: the global configuration 
        """
        ## the main training
        try:

            self.logger.info('Beginning the training loop')
            stime = time.time()

            ## build the training and valid data 
            train_data,valid_data,stable,trans = build_data(config)

            ## reranker data?
            reranker = reranker_data(config,stable.dec_map["*end*"])

            self._train(config.epochs,
                            train_data,
                            valid_data,
                            reranker,
                            config)

        except Exception,e:
            self.logger.info(e,exc_info=True)
        finally:
            self.logger.info('Finished training in %f seconds' % (time.time()-stime))

    cdef int _train(self,int epochs,
                        ParallelDataset train,
                        ParallelDataset valid,
                        RerankList rerank,
                        object config
                        ) except -1:
        """C training loop

        :param epochs: the number of epochs or iterations 
        :param source: the source data input 
        :param target: the target data input 
        :param rerank: a rerank object for evaluate model rank ability (optional)
        :param config: the global configuration 
        """
        cdef int data_point,epoch,data_size = train.size
        cdef ComputationGraph cg = self.cg
        cdef Trainer trainer = <Trainer>self.trainer

        ## training data
        cdef int[:] source,target
        cdef TransPair pair
        
        cdef Expression loss
        cdef double loss_value,epoch_loss,val_loss
        cdef double vtime,itime,vstart

        ## neural network model
        cdef Seq2SeqModel model = <Seq2SeqModel>self.model

        ## information about iteration 
        cdef int last_improvement = 0
        cdef double prev_loss = np.inf
        cdef double best_loss = np.inf
        cdef int best_score = 0

        ## dropout parameter 
        cdef double drop_out = config.dropout

        ## words observed
        cdef double tsize = float(train.target_words)
        cdef double vsize = float(valid.target_words)
        cdef double num_words

        last_backup = time.time()

        ## overall iteration
        for epoch in range(epochs):
            estart = time.time()
            epoch_loss = 0.0
            self.logger.info('Starting epoch %d' % epoch)

            ## shuffle dataset?
            train.shuffle()
            num_words = 0.0

            ## go through each data point 
            for data_point in range(data_size):

                ## reported on the ongoing progress 
                if ((data_point+1) % 10000) == 0:
                    self.logger.info('training number: %d' % data_point)

                ## interim report on validation for large datsets
                elif ((data_point+1) % 2000) == 0 and train.size > 15000:
                    ## iterim validation loss 
                    val_loss = compute_val_loss(model,valid,cg,max_check=1000)
                    self.logger.info('Val loss after %d on random 1000: %f' % (data_point,val_loss))

                    ## train perplexity
                    self.logger.info('Train perplexity %f' % np.exp(epoch_loss/num_words))

                ## backup model every 12 hours 
                if time.time() - last_backup >= 36000.0:
                    time_backup(config.dir,self.model)
                    last_backup = time.time()

                ## compute loss and back propogate
                pair = train.get_next()
                source = pair.source
                target = pair.target

                ## increment number of words
                num_words += float(target.shape[0])

                ## compute the loss 
                loss = model.get_loss(source,target,cg,drop_out=drop_out)
                loss_value = loss.value()
                loss.backward()
                epoch_loss += loss_value

                ## do online update 
                trainer.update()

            ## after epoch information 
            trainer.update_epoch(1.0)
            val_loss = 0.0
            vtime    = 0.0
            
            itime = time.time()-estart

            ## compute loss on validation
            if not valid.is_empty:
                vstart = time.time()
                val_loss = compute_val_loss(model,valid,cg)
                vtime = time.time()-vstart

                ## computer information about how well we are doing 
                if val_loss < prev_loss:
                    last_improvement = epoch

                    ## best model encountered so far?
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_score = epoch

                        ## backup the best model? 
                        if config.backup_best: 
                            backup_model(self.model,config.dir,epoch)
                            last_backup = time.time()

                ## early stopping 
                elif (epoch - last_improvement) > 3 or (epoch - best_score) > 3:
                    self.logger.info('Stopping early after %s iterations, best_epoch=%s' %\
                                         (str(epoch+1),str(best_loss)))
                    break 

                prev_loss = val_loss

            ## log information about the run 
            self.log_epoch(epoch+1,itime,epoch_loss,vtime,val_loss,tsize,vsize)

        #model.copy = False 
        ## log the best run 
        self.logger.info('Best validation run: %s' % str(best_score+1))

        ## swap the model with the best one 
        #if not best_score == (epochs-1) and backup_best:
        if config.backup_best and epochs >= 1: 
            self.model = load_best(config,best_score)

    def swap_model(self,config):
        """Swap the current model for another one 

        :param config: the global configuration 
        :param model_loc: the location of the model directory 
        :rtype: None 
        """
        model = load_new(config)
        self.logger.info('Swapping with specified model: %s....' % config.from_neural)
        self.model = model
    
    @classmethod
    def from_config(cls,config):
        """Create a Seq2SeqLearner from configuration 

        :param config: the main or global configuration 
        :param data: a data instance 
        :type data: cynet.util.DataManager
        """
        cdef Seq2SeqModel model
        cdef ParallelDataset train_data,valid_data
        cdef SymbolTable symbol_table
        cdef SparseIBMM1 lex_model,lex_model2

        ## build a lexical model (if specified)  
        if config.lex_model or config.model == 'lexattention':

            #lex_table = build_prob_lex(config)
            lex_model,lex_model2 = build_prob_lex(config)

            ## the lexical parameter tables 
            lex_table = lex_model.model_table_np()
            lex_table2 = lex_model2.model_table_np()
                
            ## its reverse because we are using reverse model 
            model_flex = lex_model.elex
            model_elex = lex_model.flex

        else:
            lex_table  = np.empty(0)
            lex_table2 = np.empty(0)

        ## build the data
        train_data,valid_data,symbol_table,trans = build_data(config)
        config.enc_vocab_size = symbol_table.enc_vocab_size
        config.dec_vocab_size = symbol_table.dec_vocab_size

        ## find copies?
        if config.copy_mechanism:
            copies = symbol_table.find_copies(partial_match=config.partial_match)
        else: copies = {}

        #constraints = build_constraints(config.model,train_data,symbol_table.dec_map["*end*"])
        nclass = NeuralModel(config.model)        
        model = <Seq2SeqModel>nclass.from_config(config,
                                                     constraints=trans,
                                                     lex=lex_table,
                                                     copies=copies,
                                                     lex2=lex_table2)

        # ## check that the ends match
        assert symbol_table.enc_map["*end*"] == model._eend
        assert symbol_table.dec_map["*end*"] == model._dend
        if config.lex_model:
            for (token,idx) in symbol_table.enc_map.items():
                if token == "*end*": continue
                assert idx == model_elex[token],"bad encoder token match"
            for (token,idx) in symbol_table.dec_map.items():
                if token == "*end*": continue
                assert idx == model_flex[token],"bad decoder token match"

        # ## find the desired trainer
        trainer = TrainerModel(config,model.model)
        #return cls(trainer,model,symbol_table)
        return cls(trainer,model,symbol_table,config)

    ## backup protocol

    def backup(self,wdir):
        """Back up the given model to file

        :param wdir: the working directory, or place to back up to
        :rtype: None 
        """
        self.logger.info('Backing up underlying neural components...')
        neural_dir = os.path.join(wdir,"neural_model")

        ## check if the directory exists, if it does than it is already backed up
        if os.path.isdir(neural_dir):
            self.logger.info('Already backed up, skipping..')
            return

        self.logger.info('making the neural directory...')
        os.mkdir(neural_dir)

        # ## back up the symbol table 
        self.stable.backup(neural_dir)

        ### backup the config
        self._config.print_to_yaml(neural_dir)
            
        ## back up the main model
        self.model.backup(neural_dir)

    @classmethod
    def load_backup(cls,config,testing=False):
        """Load a model instance from file 

        :param config: the main configuration 
        :returns: a seq2seq model instance 
        """
        cdef Seq2SeqLearner instance

        odir = config.dir
        neural_dir = os.path.join(odir,"neural_model")
        config.dir = neural_dir

        ## load the configuration
        dconfig = ConfigAttrs()
        dconfig.restore_old(neural_dir)
        #dconfig.dir = config.dir
        dconfig.dir = neural_dir

        ## load the symbol table
        #stable = SymbolTable.from_config(config)
        stable = SymbolTable.load_backup(config)
        config.enc_vocab_size = stable.enc_vocab_size
        config.dec_vocab_size = stable.dec_vocab_size

        ## find the desired class
        #nclass = NeuralModel(config.model)
        if config.from_neural and os.path.isdir(config.from_neural):
            neural_dir = config.from_neural

        nclass = NeuralModel(dconfig.model)
        model = <Seq2SeqModel>nclass.load_backup(dconfig)

        ## restore the original working directory 
        config.dir = odir

        ## find the desired trainer
        #trainer = TrainerModel(config,model.model)
        trainer = TrainerModel(dconfig,model.model)

        ## return the item
        #instance = cls(trainer,model,stable)
        instance = cls(trainer,model,stable,dconfig)
        return instance


## helper classes

cdef double sigmoid(double x):
  return 1.0 / (1.0 + np.exp(-x))


cdef class TransPair(ZubrSerializable):
    def __init__(self,source,target,idx):
        self.source = source
        self.target = target
        self._idx = idx

cdef class AttentionInfo:
    """Class for holding vector information associated with the attention mechanism"""

    def __init__(self,attention_weights,context_vector,unormalized):
        """Create an attention info instance 

        :param attention_weights: the softmax of the attention scores 
        :param context_vector: the resulting context vector
        """
        self.attention_weights = attention_weights
        self.context_vector = context_vector
        self.unormalized = unormalized

cdef class EncoderInfo:
    """Class for holding encoder information"""

    def __init__(self,list encoded,lex_probs=None,lookup={}):
        """Create an EncoderInfo instance 

        :param encoded: encoded representation of source input
        :param lex_probs: the lexical probability parameters 
        :param lookup: the matched lookup 
        """
        self.encoded = encoded
        ## add lex probs (if exists) and switch
        if lex_probs is None:
            self.lex_probs = inputTensor(np.empty(0))
            self.has_lex = False
        else:
            self.lex_probs = inputTensor(lex_probs) 
            self.has_lex = True

        ## match lookup 
        self.lookup = lookup

cdef class ParallelDataset(ZubrSerializable):
    """A class for working with parallel datasets"""

    def __init__(self,np.ndarray source,np.ndarray target,bint shuffle=True):
        """Create a ParallelDataset instance 

        :param source: the source language 
        :param target: the target language 
        :raises: ValueError
        """
        self.source = source
        self.target = target
        self._len = self.source.shape[0]
        self._shuffle = shuffle
        
        ## check that both datasets match in size
        assert self._len == self.target.shape[0],"Bad size!"
        ## dataset order
        self._dataset_order = np.array([i for i in range(self._len)])
        self._index = 0

        ## find number of target words in
        self.target_words = 0
        
        for i in range(self._len):
            self.target_words += self.target[i].shape[0]
                            
    property size:
        """Access information about the dataset size"""
        def __get__(self):
            return <int>self._len

    property shuffle:
        """Turn on and off the shuffling settings"""
        def __get__(self):
            return <bint>self._shuffle
        def __set__(self,bint new_val):
            self._shuffle = new_val

    property is_empty:
        """Deteremines if a given dataset is empty or not"""
        def __get__(self):
            return <bint>(self._len == 0)

    cdef void shuffle(self):
        """shuffle the order of the dataset (e.g. for online learning)

        :rtype: None
        """
        np.random.seed(10)
        np.random.shuffle(self._dataset_order)
        self.logger.info('Shuffled dataset order...')
        self._index = 0

    cdef TransPair get_next(self):
        """Return the next item in shuffled order

        """
        if self._index >= self._len:
            self.logger.info('Resetting iterator (should you be shuffling?)...')
            self._index = 0

        new_index = <int>self._dataset_order[self._index]
        self._index += 1
        return TransPair(self.source[new_index],self.target[new_index],new_index)

    @classmethod
    def make_empty(cls):
        """Make an empty dataset

        :returns: ParallelDataset instance without data
        """
        return cls(np.array([]),np.array([]))


cdef class RerankList(ZubrSerializable):
    """A list of items to test reranking """

    def __init__(self,sentence_ranks,rep_map,baseline):
        """Creates a rank list instance

        :param sen_list: the sentence list by identifiers 
        :param rank_map: sentences paired with rank 
        :param rep_map: the representaiton map
        """
        self.sentence_ranks = sentence_ranks
        self.rep_map        = rep_map
        self.baseline       = baseline

    property is_empty:
        """Check if this item is empty"""
        def __get__(self):
            """ 

            :rtype: bool
            """
            return self.sentence_ranks.shape[0] <= 0

    @classmethod
    def build_empty(cls):
        """Build an empty rerank item list 
        
        :returns: An empty rerank list instance 
        """
        return cls(np.empty(0),{},set())
    
cdef class SymbolTable(ZubrSerializable):
    """Hold information about the integer symbol mappings"""
    
    def __init__(self,enc_map,dec_map):
        """Creates a symbol table instance 

        :param enc_map: the encoder lexicon and word -> id map 
        :param dec_map: the decoder lexicon and word -> id map 
        :param enc_end: the <EOS> id in the encoder vocabulary 
        :param dec_end: the <EOS> id in the decoder vocabulary 
        """
        self.enc_map = enc_map
        self.dec_map = dec_map
        ## TODO: map these too lookup vectors
        
    property enc_vocab_size:
        """Get information about the encoder vocabulary size"""
        def __get__(self):
            return <int>len(self.enc_map)

    property dec_vocab_size:
        """Get information about the decoder vocabulary size"""
        def __get__(self):
            return <int>len(self.dec_map)

    def backup(self,wdir):
        """Back up the symbol table to file 

        :param wdir: the working directory 
        :rtype: None 
        """
        out = os.path.join(wdir,"symbol_map")
        np.savez_compressed(out,self.enc_map,self.dec_map)

    @classmethod
    def load_backup(cls,config):
        """Loads a symbol table from file 

        :param config: the global configuration 
        :returns: an instantiated symbol table 
        """
        out = os.path.join(config.dir,"symbol_map.npz")
        archive = np.load(out)
        enc_map = archive["arr_0"].item()
        dec_map = archive["arr_1"].item()
        return cls(enc_map,dec_map)

    ## find copies
    def find_copies(self,partial_match=True):
        """Find copies between the two different types of vocabularies 

        :param partial_match: allow partial matches to cound as copies 
        :type partial_match: bool 
        """
        copy_map = find_matches(self.enc_map,self.dec_map,partial_match)
        return copy_map

cdef class RNNResult:

    """The result of running an RNN"""
    def __init__(self,Expression prob,RNNState state,Expression attention_scores,bint copy):
        """Creates an RNNResult instance 

        :param prob: the softmax output  
        :param state: the RNN state generated
        """
        self.probs = prob
        self.state = state
        self.attention_scores = attention_scores
        self.copy = copy

cdef class HybridRNNResult(RNNResult):

    """The result of running an RNN"""
    def __init__(self,Expression prob,RNNState state,Expression attention_scores,bint copy,Expression gate):
        """Creates an RNNResult instance 

        :param prob: the softmax output  
        :param state: the RNN state generated
        """
        self.probs = prob
        self.state = state
        self.attention_scores = attention_scores
        self.copy = copy
        self.gate = gate

cdef class TransitionPair:
    """Class for representation transition constraint components"""
    
    def __init__(self,tlist,tlookup):
        """ 

        :param tlist: the list of permissible transitions 
        :type tlist: np.ndarray 
        :param tlookup: lookup list of where items are in list 
        :type tlookup: dict
        """
        self.tlist = tlist
        self.tlookup = tlookup
        
cdef class TransitionTable:
    """Stores information about allowed transitions for the constrained models"""
    
    def __init__(self,transitions):
        """Initializes a TransitionTable

        :param transition: list of transitions and lookup tables 
        """
        self.transitions = transitions

    cdef TransitionPair get_constraints(self,int indx):
        """Get constraints associated with a given prediction

        :param indx: the index of the current point or owrd in prediction 
        :returns: transition pair 
        """
        cdef list transitions = self.transitions
        cdef tuple trans_item = transitions[indx]
        
        return TransitionPair(trans_item[0],trans_item[1])

    ## backup protocol

    def backup(self,wdir):
        """Write the transition table to file  

        :param wdir: the working directory 
        """
        out = os.path.join(wdir,"transition_table")
        np.savez_compressed(out,self.transitions)
                
    @classmethod
    def load_backup(cls,config):
        """Load a backup of a TransitionTable from file

        :param config: the global configuration object 
        """
        out = os.path.join(config.dir,"transition_table.npz")
        archive = np.load(out)
        transitions = list(archive["arr_0"])
        return cls(transitions)

##
        
##

cpdef dict find_matches(emap,dmap,partial_match=True):
    """Find matches between decoder and encoder vocabulary

    :param emap: encoder map 
    :param dmap: dcoder map 
    :param partial_map: allows copies of partially matching items 
    """
    cdef int eid,fid

    ## new maps
    cdef dict matches = {},partial_matches = {}

    ## pure match
    for (eword,eid) in emap.items():
        for (fword,fid) in dmap.items():
            fword = fword.strip()
            fword = re.sub(r'\@.+$','',fword).strip()
            ## for subword cases
            fword = re.sub(r'\@\@$','',fword).strip()
            eword = eword.strip()

            ## full matches
            if eword == fword:
                matches[eid] = fid

            ## partial matches
            if eword in fword.split("_"):

                ## disallow ambiguous partial matches
                if eword in partial_matches:
                    del partial_matches[eword]
                    continue
                partial_matches[eid] = fid

    ## add unknown partial matches to match dictionary

    if partial_match:
        for (eid,fid) in partial_matches.items():
                ## add partial matches
                if eid not in matches:
                    matches[eid] = fid

    ## highly probable matches

    ## return the matched items matches
    return matches

cpdef build_prob_lex(object config):
    """Builds a probabilistic lexical model for use with the neural models
    :param config: the main configuration  
    :rtype: np.ndarray 
    :returns: lexical probability table 
    """
    cdef SymmetricWordModel sym
    cdef SparseIBMM1 etof,ftoe

    config.modeltype = "sparse_ibm1"
    config.extra_phrases = False

    ## link up the correct data
    oatraining = config.atraining

    ## switch with lex special dataset
    elex = config.atraining+"_lex.e"
    flex = config.atraining+"_lex.f"

    ## possibly backup names 
    ename = config.atraining+".e"
    fname = config.atraining+".f"
    backup_e = os.path.join(config.dir,"old_e.txt")
    backup_f = os.path.join(config.dir,"old_f.txt")

    ## initialize the model 
    sym = SymmetricWordModel.from_config(config)
    ## train the model
    sym.train(config)

    ## e->f model
    etof = sym.etof
    ftoe = sym.ftoe
    
    ## remove the alignment directories
    shutil.rmtree(os.path.join(config.dir,"alignment"))
    shutil.rmtree(os.path.join(config.dir,"alignment2"))
    config.align_dir = None
    config.sym = None
        
    #return etof.model_table_np()
    return (etof,ftoe)


cdef double compute_rerank(Seq2SeqModel model,
                               ParallelDataset data,
                               ComputationGraph cg,
                               RerankList rerank,
                               int rerank_size):
    """Evaluate the current model on reranking lists 

    :param model: the neural model 
    :param data: the validation dataset 
    :param cg: the computation graph 
    :param rerank: the list of rerank items
    """
    cdef int data_size = data.size,can_size 
    cdef np.ndarray[ndim=2,dtype=np.int32_t] ranks = rerank.sentence_ranks
    cdef int to_pick = min(rerank_size,data_size)
    cdef np.ndarray source = data.source
    cdef np.ndarray target = data.target
    cdef list points
    cdef int[:] order,candidates
    cdef double[:] can_scores
    cdef dict rep_map = rerank.rep_map
    cdef Expression loss

    points = range(data_size)
    random.seed(42)
    random.shuffle(points)
    order = np.array(points[:to_pick],dtype=np.int32)
    
    for i in range(to_pick):
        candidates = ranks[order[i]]
        can_size = candidates.shape[0]
        can_scores = np.zeros((can_size),dtype='d')

        loss = model.get_loss(source[order[i]],target[order[i]],cg)
        can_scores[0] = loss.value()

        ## find loss on each candidate 
        for j in range(1,can_size):
            loss = model.get_loss(source[order[i]],rep_map[candidates[j]],cg)
            can_scores[j] = <double>loss.value()

        ## correct 
        # if np.argmin(can_scores) == can_scores[0]:
        #     print "yes"

cdef double compute_val_loss(Seq2SeqModel model,ParallelDataset data,ComputationGraph cg,int max_check=-1):
    """Compute loss on a validation dataset given a neural model 

    :param model: the underlying model 
    :param data: the development or held out data 
    :param cg: the computation graph 
    """
    cdef np.ndarray source = data.source
    cdef np.ndarray target = data.target
    cdef int data_point,data_size = data.size
    cdef double total_loss = 0.0
    cdef int size
    cdef int[:] order
    cdef list points
    cdef Expression loss

    max_check = min(data_size,max_check)

    ## random shuffle if max_check
    if max_check == -1:
        size = data_size
        order = np.array(range(data_size),dtype=np.int32)
    else:
        size = min(max_check,data_size)
        points = range(data_size)

        ## create a random set of points to test for
        ## importantly, the seed gurantees it's always the same set (other likelihood comparison wont make sense)
        random.seed(42)
        random.shuffle(points)
        order = np.array(points[:max_check],dtype=np.int32)

    #for data_point in range(data_size):
    for data_point in range(size):
        loss = model.get_loss(source[order[data_point]],target[order[data_point]],cg)
        # loss = model.get_loss(source[data_point],target[data_point],cg)
        total_loss += <double>loss.value()

    return total_loss

## factories

MODELS = {
    "attention"    : AttentiveEncoderDecoder,
    "cattention"   : ConstrainedAttention,
    "lexattention" : AttentionLexModel,
}

TRAINERS = {
    "sgd"      : SimpleSGDTrainer,
    "momentum" : MomentumSGDTrainer,
    "adagrad"  : AdagradTrainer,
    "adam"     : AdamTrainer,
}

cpdef NeuralModel(ntype):
    """Factory method for getting a neural model

    :param ntype: the type of neural model desired 
    :raises: ValueError
    """
    nclass = MODELS.get(ntype,None)
    if nclass is None:
        raise ValueError('Unknown neural model: %s' % ntype)
    return nclass

def TrainerModel(config,model):
    """Factory method for selecting a particular trainer 

    :param ttype: the type of trainer to use 
    """
    tname = config.trainer
    tclass = TRAINERS.get(tname)
    if tclass is None:
        raise ValueError("Unknown trainer model: %s" % tname)

    if tname == "adagrad":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay,eps=config.epsilon)
    elif tname == "sgd":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay)
    elif tname == "momentum":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay,mom=config.momentum)
    elif tname == "adam":
        trainer = tclass(model,alpha=config.lrate,beta_1=config.beta_1,eps=config.eps,edecay=config.weight_decay)
    return trainer


def params():
    """The parameters for running this neural stuff 

    """
    from zubr.Alignment import params as a_params
    from zubr.SymmetricAlignment import params as s_params

    aligner_group,aligner_params = a_params()
    s_group,saligner_params = s_params()

    options = [
        ("--enc_rnn_layers","enc_rnn_layers",1,int,
         "Number of layers to use in encoder RNN [default=1]","Seq2Seq"),
        ("--dec_rnn_layers","dec_rnn_layers",1,int,
         "Number of layers to use in decder RNN [default=1]","Seq2Seq"),
        ("--embedding_size","embedding_size",10,int,
         "The size of the embeddings [default=10]","Seq2Seq"),
        ("--dec_state_size","dec_state_size",64,int,
         "The size of the decoder state size [default=64]","Seq2Seq"),
        ("--enc_state_size","enc_state_size",64,int,
         "The size of the encoder state size [default=64]","Seq2Seq"),
        ("--model","model","attention","str",
         "The type of sequence2sequence model to use [default='simple']","Seq2Seq"),
        ("--test_v","test_v",False,"bool",
         "Test model on validation after loading (for debugging) [default=False]","Seq2Seq"),
        ("--backup_best","backup_best",False,"bool",
         "When training, back up the best model [default=False]","Seq2Seq"),
        ("--lex_model","lex_model",False,"bool",
         "Use a probabilistic lexicon when training/decoding [default=False]","Seq2Seq"),
        ("--lex_param","lex_param",0.001,"float",
         "The lexical paramter for model biasing [default=0.001]","Seq2Seq"),
        ("--interp_param","interp_param",0.0,"float",
         "The interpolation parameter for joint lex/neural models [default=0.0]","Seq2Seq"),
        ("--interpolate","interpolate",False,"bool",
         "Use linear interpolation of neural and lex model [default=False]","Seq2Seq"),
        ("--copy_mechanism","copy_mechanism",False,"bool",
         "Use the copy model [default=False]","Seq2Seq"),
        ("--run_copy","run_copy",0,"int",
         "Run the copy mechanism only after a certain number of iterations [default=0]","Seq2Seq"),
        ("--partial_match","partial_match",False,"bool",
         "Add partial matches to copy list [default=False]","Seq2Seq"),
        ("--rerank","rerank",False,"bool",
         "Check progress by reranking baseline output [default=False]","Seq2Seq"),
        ("--rerank_size","rerank",500,int,
         "The size of the rerank list [default=500]","Seq2Seq"),                                                      
    ]

    group = {"Seq2Seq" : "General settings for building Seq2Seq Models"}
    group.update(s_group)
    group.update(aligner_group)
    options += aligner_params
    options += saligner_params
    return (group,options)


def run_seq2seq(config):
    """Main execution point for running a seq2seq model 
    
    :param config: the global configuration 
    """
    
    try: 
        learner = Seq2SeqLearner.from_config(config)

        ## train the model 
        learner.train(config)

        ## backup the model 
        learner.backup(config.dir)

    except Exception,e:
        traceback.print_exc(file=sys.stderr)
