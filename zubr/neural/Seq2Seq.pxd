from zubr.ZubrClass cimport ZubrSerializable
cimport numpy as np
import numpy as np

from zubr.neural._dynet cimport (
    Expression,
    Trainer,
    ComputationGraph,
    ParameterCollection,
    LookupParameters,
    LSTMBuilder,
    Parameters,
    RNNState,
    log,
)


cdef class Seq2SeqModel(ZubrSerializable):
    cdef ParameterCollection model
    cdef LookupParameters enc_embeddings
    cdef LookupParameters dec_embeddings
    cdef Parameters output_w,output_b
    ## methods 
    cdef Expression get_loss(self, int[:] x, int[:] z,ComputationGraph cg, double drop_out=?)
    cdef list _embed_x(self,int[:] x,ComputationGraph cg)
    cdef list _embed_z(self,int[:]z,ComputationGraph cg)
    cdef list _run_enc_rnn(self,RNNState init_state,list input_vecs)
    cdef Expression _get_probs(self,Expression rnn_output)
    cdef EncoderInfo encode_input(self,int[:] x,ComputationGraph cg)
    ### EOS positions
    cdef int _eend,_dend
    cdef bint copy
    cdef dict copies 

cdef class RNNSeq2Seq(Seq2SeqModel):
    cdef LSTMBuilder enc_rnn, dec_rnn
    
cdef class EncoderDecoder(RNNSeq2Seq):
    cdef EncoderInfo _encode_string(self,list embeddings,int[:] einput)
    cdef int enc_state_size
    ### methods 
    cdef RNNState get_dec_init(self,ComputationGraph cg)
    cdef RNNState append_state(self,RNNState s,EncoderInfo e, Expression imatrix,int identifier,ComputationGraph cg)
    cdef RNNResult get_dec_distr(self,RNNState s,EncoderInfo e,Expression imatrix,Expression last_embed)
    cdef RNNResult get_dec_distr_scratch(self,RNNState s,EncoderInfo e,
                                             Expression imatrix,int indx,ComputationGraph cg)
    cdef Expression get_init_embedding(self,ComputationGraph cg)
    cdef Expression get_dec_embed(self,int idx,ComputationGraph cg)
    cdef np.ndarray lex_model
    cdef bint interpolate #,copy
    cdef double lex_param,interp_param

cdef class AttentionModel(EncoderDecoder):
    cdef Parameters attention_w1,attention_w2,attention_v
    cdef Expression _attend(self,list input_vectors, RNNState state)

cdef class BiLSTMAttention(AttentionModel):
    cdef LSTMBuilder enc_bwd_rnn
    cdef AttentionInfo _bi_attend(self,Expression input_matrix,RNNState s,Expression w1dt)

cdef class AttentiveEncoderDecoder(BiLSTMAttention):
    cdef Parameters output_final,final_bias

cdef class AttentiveEncoderDecoderMore(AttentiveEncoderDecoder):
    pass

cdef class ConstrainedAttention(AttentiveEncoderDecoder):
    cdef TransitionTable transitions


## hybrid model

cdef class AttentionLexModel(ConstrainedAttention):
    cdef np.ndarray lex_model2
    cdef Parameters gate,gate_bias

## learner

cdef class Seq2SeqLearner(ZubrSerializable):
    cdef public ComputationGraph cg
    cdef public Seq2SeqModel model
    cdef public SymbolTable stable
    cdef object _config 
    cdef Trainer trainer
    ## method
    cpdef void train(self,config)
    cdef int _train(self,int epochs,ParallelDataset train, ParallelDataset valid,RerankList rerank,
                        object config) except -1

## helper classes

cdef class TransPair(ZubrSerializable):
    cdef np.ndarray source,target
    cdef int _idx

cdef class ParallelDataset(ZubrSerializable):
    cdef np.ndarray source,target,_dataset_order
    cdef int _len,_index,target_words
    cdef bint _shuffle
    cdef TransPair get_next(self)
    cdef void shuffle(self)
    
cdef class RerankList(ZubrSerializable):
    cdef np.ndarray sentence_ranks
    cdef dict rep_map
    cdef set baseline 

cdef class SymbolTable(ZubrSerializable):
    cdef public dict enc_map
    cdef public dict dec_map
    cdef int _vocab_size

cdef class RNNResult:
    cdef Expression probs,attention_scores
    cdef RNNState state
    cdef bint copy 

cdef class HybridRNNResult(RNNResult):
    cdef Expression gate
    
cdef class TransitionPair:
    cdef np.ndarray tlist
    cdef dict tlookup

cdef class TransitionTable:
    cdef list transitions
    cdef TransitionPair get_constraints(self,int indx)

cdef class AttentionInfo:
    cdef Expression attention_weights,context_vector,unormalized

cdef class EncoderInfo:
    cdef list encoded
    cdef Expression lex_probs
    cdef bint has_lex
    cdef dict lookup
    
cpdef NeuralModel(ntype)
