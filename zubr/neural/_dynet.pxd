

## basically this is where all the c++ dynet stuff gets called

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

ctypedef float real

cdef extern from "dynet/init.h" namespace "dynet":
    cdef cppclass CDynetParams "dynet::DynetParams":
        unsigned random_seed
        string mem_descriptor
        float weight_decay
        int autobatch
        int autobatch_debug
        bool shared_parameters
        bool ngpus_requested
        bool ids_requested
        int requested_gpus
        vector[int] gpu_mask
    cdef CDynetParams extract_dynet_params(int& argc, char**& argv, bool shared_parameters)
    cdef void initialize(CDynetParams params)
    cdef void initialize(int& argc, char **& argv, bool shared_parameters)

cdef extern from "dynet/dim.h" namespace "dynet":
    cdef cppclass CDim "dynet::Dim":
        CDim() except +
        CDim(vector[long]& ds) except +
        CDim(vector[long]& ds, unsigned int bs) except +
        int size()
        unsigned int batch_elems()
        int sum_dims()
        CDim truncate()
        void resize(unsigned i)
        int ndims()
        int rows()
        int cols()
        unsigned operator[](unsigned i)
        void set(unsigned i, unsigned s)
        int size(unsigned i)
        CDim transpose()

cdef extern from "dynet/tensor.h" namespace "dynet":
    cdef cppclass CTensor "dynet::Tensor": 
        CDim d
        float* v
        pass
    float c_as_scalar "dynet::as_scalar" (CTensor& t)
    vector[float] c_as_vector "dynet::as_vector" (CTensor& t)

cdef extern from "dynet/tensor.h" namespace "dynet":
    cdef cppclass CIndexTensor "dynet::IndexTensor": 
        CDim d
        pass
    vector[ptrdiff_t] c_index_tensor_as_vector "dynet::as_vector" (CIndexTensor& t)
    cdef cppclass CTensorTools "dynet::TensorTools":
        @staticmethod
        CIndexTensor argmax(CTensor& t, unsigned dim, unsigned num) 
        @staticmethod
        CIndexTensor categorical_sample_log_prob(CTensor& t, unsigned dim, unsigned num) 

cdef extern from "dynet/model.h" namespace "dynet":
    cdef cppclass CParameterStorage "dynet::ParameterStorage":
        CParameterStorage()
        CTensor values
        CTensor g
        CDim dim
        void clip(float left, float right)

    cdef cppclass CLookupParameterStorage "dynet::LookupParameterStorage":
        CLookupParameterStorage()
        vector[CTensor] values
        vector[CTensor] grads
        CDim dim
        CDim all_dim

    cdef cppclass CParameters "dynet::Parameter":
        CParameters()
        CParameterStorage& get_storage()
        void zero()
        void set_updated(bool b)
        bool is_updated()
        void scale(float s)
        void scale_gradient(float s)
        void clip_inplace(float left, float right)
        string get_fullname()

    cdef cppclass CLookupParameters "dynet::LookupParameter":
        CLookupParameters()
        CLookupParameterStorage& get_storage()
        CDim dim
        void initialize(unsigned index, const vector[float]& val)
        void zero()
        void set_updated(bool b)
        bool is_updated()
        void scale(float s)
        void scale_gradient(float s)
        string get_fullname()

    cdef cppclass CModel "dynet::ParameterCollection":
        CModel()
        #float gradient_l2_norm() const
        CParameters add_parameters(CDim& d)
        CParameters add_parameters(CDim& d, CParameterInit initializer, string name)
        #CLookupParameters add_lookup_parameters(unsigned n, const CDim& d)
        CLookupParameters add_lookup_parameters(unsigned n, const CDim& d, CParameterInit initializer, string name)
        vector[CParameterStorage] parameters_list()
        CModel add_subcollection(string name)
        string get_fullname()

cdef extern from "dynet/io.h" namespace "dynet":
    cdef cppclass CTextFileSaver "dynet::TextFileSaver":
        CTextFileSaver(string filename, bool append)
        void save(CModel model, string & key)
        void save(CParameters param, string & key)
        void save(CLookupParameters param, string & key)

    cdef cppclass CTextFileLoader "dynet::TextFileLoader":
        CTextFileLoader(string filename)
        void populate(CModel & model, string key)
        void populate(CParameters & param, string key) except +
        void populate(CLookupParameters & param, string key) except +
        CParameters load_param(CModel & model, string key) except +
        CLookupParameters load_lookup_param(CModel & model, string key) except +


cdef extern from "dynet/param-init.h" namespace "dynet":
    cdef cppclass CParameterInit "dynet::ParameterInit":
        pass

    cdef cppclass CParameterInitNormal "dynet::ParameterInitNormal" (CParameterInit):
        CParameterInitNormal(float m, float v) # m = 0, v=1

    cdef cppclass CParameterInitUniform "dynet::ParameterInitUniform" (CParameterInit):
        CParameterInitUniform(float scale)

    cdef cppclass CParameterInitConst "dynet::ParameterInitConst" (CParameterInit):
        CParameterInitConst(float c)

    cdef cppclass CParameterInitIdentity "dynet::ParameterInitIdentity" (CParameterInit):
        CParameterInitIdentity()

    cdef cppclass CParameterInitGlorot "dynet::ParameterInitGlorot" (CParameterInit):
        CParameterInitGlorot(bool is_lookup, float gain) # is_lookup = False

    cdef cppclass CParameterInitSaxe "dynet::ParameterInitSaxe" (CParameterInit):
        CParameterInitSaxe(float scale)

    cdef cppclass CParameterInitFromFile "dynet::ParameterInitFromFile" (CParameterInit):
        CParameterInitFromFile(string filename)

    cdef cppclass CParameterInitFromVector "dynet::ParameterInitFromVector" (CParameterInit):
        CParameterInitFromVector(vector[float] void)


cdef extern from "dynet/dynet.h" namespace "dynet":
    ctypedef unsigned VariableIndex

    cdef cppclass CComputationGraph "dynet::ComputationGraph":
        CComputationGraph() except +
        CComputationGraph(bool autobatched) except +
        # Inputs
        VariableIndex add_input(real s) except +
        VariableIndex add_input(const real* ps) except +
        VariableIndex add_input(const CDim& d, const vector[float]* pdata) except +

        # Parameters
        VariableIndex add_parameters(CParameters* p) except +
        VariableIndex add_lookup(CLookupParameters* p, const unsigned* pindex) except +
        VariableIndex add_lookup(CLookupParameters* p, unsigned index) except +
        VariableIndex add_const_lookup(CLookupParameters* p, const unsigned* pindex) except +
        VariableIndex add_const_lookup(CLookupParameters* p, unsigned index) except +
        
        const CTensor& forward(VariableIndex index) except +
        const CTensor& incremental_forward(VariableIndex index) except +
        const CTensor& get_value(VariableIndex i) except +
        void invalidate()
        void backward(VariableIndex i, bool full)

        # checkpointing
        void checkpoint()
        void revert()

        # immediate computation
        void set_immediate_compute(bool ic)
        void set_check_validity(bool cv)

        void print_graphviz() const

cdef extern from "dynet/training.h" namespace "dynet":
    cdef cppclass CTrainer "dynet::Trainer":
        CTrainer(CModel& m, float e0, float edecay) # TODO removed lam, update docs.
        float clip_threshold
        bool clipping_enabled
        bool sparse_updates_enabled
        void update(float s) except +
        #void update(vector[unsigned]& uparam, vector[unsigned]& ulookup, float s) except +
        void update_epoch(float r)
        void status()


    cdef cppclass CSimpleSGDTrainer "dynet::SimpleSGDTrainer" (CTrainer):
        CSimpleSGDTrainer(CModel& m, float e0, float edecay) # TODO removed lam, update docs.

    cdef cppclass CCyclicalSGDTrainer "dynet::CyclicalSGDTrainer" (CTrainer):
        CCyclicalSGDTrainer(CModel& m, float e0_min, float e0_max, float step_size, float gamma, float edecay) # TODO removed lam, update docs.

    cdef cppclass CMomentumSGDTrainer "dynet::MomentumSGDTrainer" (CTrainer):
        CMomentumSGDTrainer(CModel& m, float e0, float mom, float edecay) # TODO removed lam, update docs

    cdef cppclass CAdagradTrainer "dynet::AdagradTrainer" (CTrainer):
        CAdagradTrainer(CModel& m, float e0, float eps, float edecay) # TODO removed lam, update docs

    cdef cppclass CAdadeltaTrainer "dynet::AdadeltaTrainer" (CTrainer):
        CAdadeltaTrainer(CModel& m, float eps, float rho, float edecay) # TODO removed lam, update docs

    cdef cppclass CRMSPropTrainer "dynet::RMSPropTrainer" (CTrainer):
        CRMSPropTrainer(CModel& m, float e0, float eps, float rho, float edecay) # TODO removed lam, update docs

    cdef cppclass CAdamTrainer "dynet::AdamTrainer" (CTrainer):
        CAdamTrainer(CModel& m, float alpha, float beta_1, float beta_2, float eps, float edecay) # TODO removed lam, update docs


cdef extern from "dynet/expr.h" namespace "dynet":
    cdef cppclass CExpression "dynet::Expression":
        CExpression()
        CExpression(CComputationGraph *pg, VariableIndex i)
        CComputationGraph *pg
        long i
        CDim dim() except +
        bool is_stale()
        const CTensor& gradient() except +
    #CExpression c_input "dynet::input" (CComputationGraph& g, float s)   #
    CExpression c_input "dynet::input" (CComputationGraph& g, float *ps) except + #
    CExpression c_input "dynet::input" (CComputationGraph& g, CDim& d, vector[float]* pdata) except +
    CExpression c_input "dynet::input" (CComputationGraph& g, CDim& d, vector[unsigned]& ids, vector[float]& data, float defdata) except +
    CExpression c_parameter "dynet::parameter" (CComputationGraph& g, CParameters p) except + #
    CExpression c_parameter "dynet::parameter" (CComputationGraph& g, CLookupParameters p) except + #
    CExpression c_const_parameter "dynet::const_parameter" (CComputationGraph& g, CParameters p) except + #
    CExpression c_const_parameter "dynet::const_parameter" (CComputationGraph& g, CLookupParameters p) except + #
    #CExpression c_lookup "dynet::lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index) except +   #
    CExpression c_lookup "dynet::lookup" (CComputationGraph& g, CLookupParameters p, unsigned* pindex) except + #
    CExpression c_lookup "dynet::lookup" (CComputationGraph& g, CLookupParameters p, vector[unsigned]* pindices) except + #
    #CExpression c_const_lookup "dynet::const_lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index) except +   #
    CExpression c_const_lookup "dynet::const_lookup" (CComputationGraph& g, CLookupParameters p, unsigned* pindex) except + #
    CExpression c_const_lookup "dynet::const_lookup" (CComputationGraph& g, CLookupParameters p, vector[unsigned]* pindices) except + #
    CExpression c_zeroes "dynet::zeroes" (CComputationGraph& g, CDim& d) except + #
    CExpression c_random_normal "dynet::random_normal" (CComputationGraph& g, CDim& d) except + #
    CExpression c_random_bernoulli "dynet::random_bernoulli" (CComputationGraph& g, CDim& d, float p, float scale) except +
    CExpression c_random_uniform "dynet::random_uniform" (CComputationGraph& g, CDim& d, float left, float right) except + #
    CExpression c_random_gumbel "dynet::random_gumbel" (CComputationGraph& g, CDim& d, float left, float right) except + #

    # identity function, but derivative is not propagated through it
    CExpression c_nobackprop "dynet::nobackprop" (CExpression& x) except + #
    # identity function, but derivative takes negative as propagated through it
    CExpression c_flip_gradient "dynet::flip_gradient" (CExpression& x) except + #
    
    CExpression c_op_neg "dynet::operator-" (CExpression& x) except + #
    CExpression c_op_add "dynet::operator+" (CExpression& x, CExpression& y) except + #
    CExpression c_op_scalar_add "dynet::operator+" (CExpression& x, float y) except + #
    CExpression c_op_mul "dynet::operator*" (CExpression& x, CExpression& y) except + #
    CExpression c_op_scalar_mul "dynet::operator*" (CExpression& x, float y) except + #
    CExpression c_op_scalar_div "dynet::operator/" (CExpression& x, float y) except + #
    CExpression c_op_scalar_sub "dynet::operator-" (float y, CExpression& x) except + #

    CExpression c_bmax "dynet::max" (CExpression& x, CExpression& y) except + #
    CExpression c_bmin "dynet::min" (CExpression& x, CExpression& y) except + #

    CExpression c_cdiv "dynet::cdiv" (CExpression& x, CExpression& y) except + #
    CExpression c_cmult "dynet::cmult" (CExpression& x, CExpression& y) except + #
    CExpression c_colwise_add "dynet::colwise_add" (CExpression& x, CExpression& bias) except + #

    CExpression c_tanh "dynet::tanh" (CExpression& x) except + #
    CExpression c_exp "dynet::exp" (CExpression& x) except + #
    CExpression c_square "dynet::square" (CExpression& x) except + #
    CExpression c_sqrt "dynet::sqrt" (CExpression& x) except + #
    CExpression c_abs "dynet::abs" (CExpression& x) except + #
    CExpression c_erf "dynet::erf" (CExpression& x) except + #
    CExpression c_cube "dynet::cube" (CExpression& x) except + #
    CExpression c_log "dynet::log" (CExpression& x) except + #
    CExpression c_lgamma "dynet::lgamma" (CExpression& x) except + #
    CExpression c_logistic "dynet::logistic" (CExpression& x) except + #
    CExpression c_rectify "dynet::rectify" (CExpression& x) except + #
    CExpression c_hinge "dynet::hinge" (CExpression& x, unsigned index, float m) except + #
    CExpression c_hinge "dynet::hinge" (CExpression& x, vector[unsigned] vs, float m) except + #
    CExpression c_log_softmax "dynet::log_softmax" (CExpression& x) except + #
    CExpression c_log_softmax "dynet::log_softmax" (CExpression& x, vector[unsigned]& restriction) except + #?
    CExpression c_softmax "dynet::softmax" (CExpression& x) except + #
    CExpression c_sparsemax "dynet::sparsemax" (CExpression& x) except + #
    CExpression c_softsign "dynet::softsign" (CExpression& x) except + #
    CExpression c_pow "dynet::pow" (CExpression& x, CExpression& y) except + #
    CExpression c_bmin "dynet::min" (CExpression& x, CExpression& y) except + #
    CExpression c_bmax "dynet::max" (CExpression& x, CExpression& y) except + #
    CExpression c_noise "dynet::noise" (CExpression& x, float stddev) except + #
    CExpression c_dropout "dynet::dropout" (CExpression& x, float p) except + #
    CExpression c_dropout_batch "dynet::dropout_batch" (CExpression& x, float p) except + #
    CExpression c_dropout_dim "dynet::dropout_dim" (CExpression& x, unsigned d, float p) except + #
    CExpression c_block_dropout "dynet::block_dropout" (CExpression& x, float p) except + #

    CExpression c_reshape "dynet::reshape" (CExpression& x, CDim& d) except + #?
    CExpression c_transpose "dynet::transpose" (CExpression& x, vector[unsigned]& dims) except + #

    CExpression c_affine_transform "dynet::affine_transform" (const vector[CExpression]& xs) except +

    CExpression c_inverse "dynet::inverse" (CExpression& x) except + #
    CExpression c_logdet "dynet::logdet" (CExpression& x) except + #
    CExpression c_trace_of_product "dynet::trace_of_product" (CExpression& x, CExpression& y) except +;

    CExpression c_dot_product "dynet::dot_product" (CExpression& x, CExpression& y) except + #
    CExpression c_squared_distance "dynet::squared_distance" (CExpression& x, CExpression& y) except + #
    CExpression c_squared_norm "dynet::squared_norm" (CExpression& x) except + #
    CExpression c_l2_norm "dynet::l2_norm" (CExpression& x) except + #
    CExpression c_huber_distance "dynet::huber_distance" (CExpression& x, CExpression& y, float c) except + #
    CExpression c_l1_distance "dynet::l1_distance" (CExpression& x, CExpression& y) except + #
    CExpression c_binary_log_loss "dynet::binary_log_loss" (CExpression& x, CExpression& y) except + #
    CExpression c_pairwise_rank_loss "dynet::pairwise_rank_loss" (CExpression& x, CExpression& y, float m) except + #
    CExpression c_poisson_loss "dynet::poisson_loss" (CExpression& x, unsigned y) except +

    #CExpression c_conv1d_narrow "dynet::conv1d_narrow" (CExpression& x, CExpression& f) except + #
    #CExpression c_conv1d_wide "dynet::conv1d_wide" (CExpression& x, CExpression& f) except + #
    CExpression c_filter1d_narrow "dynet::filter1d_narrow" (CExpression& x, CExpression& f) except + #
    CExpression c_kmax_pooling "dynet::kmax_pooling" (CExpression& x, unsigned k, unsigned d) except + #
    CExpression c_fold_rows "dynet::fold_rows" (CExpression& x, unsigned nrows) except + #
    CExpression c_sum_cols "dynet::sum_cols" (CExpression& x) except +               #
    CExpression c_kmh_ngram "dynet::kmh_ngram" (CExpression& x, unsigned n) except + #
    CExpression c_conv2d "dynet::conv2d" (CExpression& x, CExpression& f, vector[unsigned] stride, bool is_valid) except + #
    CExpression c_conv2d "dynet::conv2d" (CExpression& x, CExpression& f, CExpression& b, vector[unsigned] stride, bool is_valid) except + #
    CExpression c_maxpooling2d "dynet::maxpooling2d" (CExpression& x, vector[unsigned] ksize, vector[unsigned] stride, bool is_valid) except + #

    CExpression c_sum_batches "dynet::sum_batches" (CExpression& x) except +
    CExpression c_sum_elems "dynet::sum_elems" (CExpression& x) except +
    CExpression c_moment_batches "dynet::moment_batches" (CExpression& x, unsigned r) except +
    CExpression c_moment_elems "dynet::moment_elems" (CExpression& x, unsigned r) except +
    CExpression c_moment_dim "dynet::moment_dim" (CExpression& x, unsigned d, unsigned r) except +
    CExpression c_mean_elems "dynet::mean_elems" (CExpression& x) except +
    CExpression c_mean_batches "dynet::mean_batches" (CExpression& x) except +
    CExpression c_mean_dim "dynet::mean_dim" (CExpression& x, unsigned d) except +
    CExpression c_std_dim "dynet::std_dim" (CExpression& x, unsigned d) except +
    CExpression c_std_elems "dynet::std_elems" (CExpression& x) except +
    CExpression c_std_batches "dynet::std_batches" (CExpression& x) except +

    #CExpression c_pick "dynet::pick" (CExpression& x, unsigned v) except +   #
    CExpression c_select_rows "dynet::select_rows" (CExpression& x, vector[unsigned] rs) except +
    CExpression c_select_cols "dynet::select_cols" (CExpression& x, vector[unsigned] cs) except +
    CExpression c_pick "dynet::pick" (CExpression& x, unsigned* pv, unsigned d) except + #
    CExpression c_pick "dynet::pick" (CExpression& x, vector[unsigned]* pv, unsigned d) except + #
    CExpression c_pick_range "dynet::pick_range" (CExpression& x, unsigned v, unsigned u, unsigned d) except + #

    CExpression c_pick_batch_elems "dynet::pick_batch_elems" (CExpression& x, vector[unsigned] vs) except + #
    CExpression c_pick_batch_elem "dynet::pick_batch_elem" (CExpression& x, unsigned v) except + #
    CExpression c_pickneglogsoftmax "dynet::pickneglogsoftmax" (CExpression& x, unsigned v) except + #
    CExpression c_pickneglogsoftmax "dynet::pickneglogsoftmax" (CExpression& x, vector[unsigned] vs) except + #

    CExpression c_contract3d_1d "dynet::contract3d_1d" (CExpression& x, CExpression& y) except + #
    CExpression c_contract3d_1d "dynet::contract3d_1d" (CExpression& x, CExpression& y, CExpression& b) except + #
    CExpression c_contract3d_1d_1d "dynet::contract3d_1d_1d" (CExpression& x, CExpression& y, CExpression& z) except + #
    CExpression c_contract3d_1d_1d "dynet::contract3d_1d_1d" (CExpression& x, CExpression& y, CExpression& z, CExpression& b) except + #

    CExpression c_elu "dynet::elu" (CExpression& x, float alpha) except + #
    CExpression c_selu "dynet::selu" (CExpression& x) except + #
    
    # expecting a vector of CExpression
    CExpression c_average     "dynet::average" (vector[CExpression]& xs) except +
    CExpression c_concat_cols "dynet::concatenate_cols" (vector[CExpression]& xs) except +
    CExpression c_concat      "dynet::concatenate" (vector[CExpression]& xs, unsigned d) except +
    CExpression c_concat_to_batch      "dynet::concatenate_to_batch" (vector[CExpression]& xs) except +

    CExpression c_sum            "dynet::sum" (vector[CExpression]& xs) except +
    CExpression c_max            "dynet::vmax" (vector[CExpression]& xs) except +
    CExpression c_logsumexp      "dynet::logsumexp" (vector[CExpression]& xs) except +

    CExpression c_max_dim "dynet::max_dim" (CExpression& x, unsigned d) except + #
    CExpression c_min_dim "dynet::min_dim" (CExpression& x, unsigned d) except + #

    CExpression c_layer_norm "dynet::layer_norm" (CExpression& x, CExpression& g, CExpression& b) except + #
    CExpression c_weight_norm "dynet::weight_norm" (CExpression& w, CExpression& g) except + #

cdef extern from "dynet/rnn.h" namespace "dynet":
    cdef cppclass CRNNPointer "dynet::RNNPointer":
        CRNNPointer()
        CRNNPointer(int i)

    cdef cppclass CRNNBuilder "dynet::RNNBuilder":
        void new_graph(CComputationGraph &cg, bool update)
        void start_new_sequence(vector[CExpression] ces)
        CExpression add_input(CExpression &x) except +
        CExpression add_input(CRNNPointer prev, CExpression &x) except +
        CExpression set_h(CRNNPointer prev, vector[CExpression] ces)
        CExpression set_s(CRNNPointer prev, vector[CExpression] ces)
        void rewind_one_step()
        CExpression back()
        vector[CExpression] final_h()
        vector[CExpression] final_s()
        vector[CExpression] get_h(CRNNPointer i)
        vector[CExpression] get_s(CRNNPointer i)
        CRNNPointer state()
        void set_dropout(float f)
        void disable_dropout()
        CModel get_parameter_collection()

cdef extern from "dynet/rnn.h" namespace "dynet":
    cdef cppclass CSimpleRNNBuilder  "dynet::SimpleRNNBuilder" (CRNNBuilder):
        CSimpleRNNBuilder()
        CSimpleRNNBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model)

        vector[vector[CParameters]] params
        vector[vector[CExpression]] param_vars


cdef extern from "dynet/gru.h" namespace "dynet":
    cdef cppclass CGRUBuilder "dynet::GRUBuilder" (CRNNBuilder):
        CGRUBuilder()
        CGRUBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model)

        vector[vector[CParameters]] params
        vector[vector[CExpression]] param_vars

cdef extern from "dynet/lstm.h" namespace "dynet":
    cdef cppclass CLSTMBuilder "dynet::LSTMBuilder" (CRNNBuilder):
        CLSTMBuilder()
        CLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model)

        vector[vector[CParameters]] params
        vector[vector[CExpression]] param_vars

    cdef cppclass CVanillaLSTMBuilder "dynet::VanillaLSTMBuilder" (CRNNBuilder):
        CVanillaLSTMBuilder()
        CVanillaLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model, bool ln_lstm)
        void set_dropout(float d, float d_r)
        void set_dropout_masks(unsigned batch_size)

        vector[vector[CParameters]] params
        vector[vector[CExpression]] param_vars

cdef extern from "dynet/fast-lstm.h" namespace "dynet":
    cdef cppclass CFastLSTMBuilder "dynet::FastLSTMBuilder" (CRNNBuilder):
        CFastLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model)

        vector[vector[CParameters]] params
        vector[vector[CExpression]] param_vars

cdef class ComputationGraph:
    cdef CComputationGraph *thisptr,
    cdef list _inputs
    cdef int _cg_version
    cpdef renew(self, immediate_compute=?, check_validity=?, autobatching=?)
    cpdef version(self)
    cpdef forward_scalar(self, VariableIndex index)
    cpdef inc_forward_scalar(self, VariableIndex index)
    cpdef forward_vec(self, VariableIndex index)
    cpdef inc_forward_vec(self, VariableIndex index)
    cpdef forward(self, VariableIndex index)
    cpdef inc_forward(self, VariableIndex index)
    cpdef backward(self, VariableIndex index, bool full=?)
    cpdef print_graphviz(self)
    cpdef void checkpoint(self)
    cpdef void revert(self)
    cdef inputValue(self, float v =?)
    cdef inputVector(self, int dim)
    cdef inputVectorLiteral(self, vector[float] v)
    cdef inputMatrix(self, int d1, int d2)
    cdef lookup(self, LookupParameters p, unsigned v =?, update=?)
    cdef lookup_batch(self, LookupParameters p, vector[unsigned] vs, update=?)
    cdef outputPicker(self, Expression e, unsigned v=?, unsigned dim=?)
    cdef outputBatchPicker(self, Expression e, vector[unsigned] vs, unsigned dim=?)
        
cdef class Expression:
    cdef VariableIndex vindex
    cdef int cg_version
    ## methods
    cdef inline ComputationGraph cg(self)
    cdef inline CComputationGraph* cgp(self)
    cdef CExpression c(self)
    cpdef scalar_value(self, recalculate=?)
    cpdef vec_value(self, recalculate=?)
    cpdef npvalue(self, recalculate=?)
    cpdef tensor_value(self, recalculate=?)
    cpdef value(self, recalculate=?)
    cpdef gradient(self)
    cpdef forward(self, recalculate=?)
    cpdef backward(self, bool full=?)
    @staticmethod
    cdef Expression from_cexpr(int cgv, CExpression cexpr)


cdef class LookupParameters:
    cdef CLookupParameters thisptr
    cdef int _version
    cdef Expression _expr
    @staticmethod
    cdef wrap_ptr(CLookupParameters ptr)
    cpdef init_from_array(self, arr)
    cpdef shape(self)
    cpdef batch(self, vector[unsigned] i)
    cpdef init_row(self, unsigned i, vector[float] row)
    cpdef as_array(self)
    cpdef grad_as_array(self)
    cpdef scale(self,float s)
    cpdef scale_gradient(self,float s)
    cpdef Expression expr(self,bool update=?)
    cpdef zero(self)
    cpdef bool is_updated(self)
    cpdef set_updated(self, bool b)
    cpdef name(self)

cdef class Parameters:
    cdef CParameters thisptr
    cdef int _version
    cdef Expression _expr
    cdef int _const_version
    cdef Expression _const_expr
    @staticmethod
    cdef wrap_ptr(CParameters ptr)
    cpdef shape(self)
    cpdef as_array(self)
    cpdef grad_as_array(self)
    cpdef clip_inplace(self, float left, float right)
    cpdef set_value(self, arr)
    cpdef zero(self)
    cpdef scale(self,float s)
    cpdef scale_gradient(self,float s)
    cpdef bool is_updated(self)
    cpdef set_updated(self, bool b)
    cpdef name(self)
    cpdef Expression expr(self, bool update=?)

cdef class PyInitializer:
    cdef CParameterInit *initializer

cdef class ParameterCollection:
    cdef CModel thisptr
    @staticmethod
    cdef wrap(CModel m)
    cpdef load_param(self, fname, key)
    cpdef load_lookup_param(self, fname, key)
    cpdef pl(self)
    cpdef lookup_parameters_from_numpy(self, array, string name=?)
    cpdef add_parameters(self, dim, PyInitializer init=?, string name=?)
    cpdef add_lookup_parameters(self, dim, PyInitializer init=?, string name=?)
    cpdef add_subcollection(self, name=?)
    cpdef name(self)
    cpdef parameters_from_numpy(self, array,string name=?)
    
cdef class Tensor:
    cdef CTensor t
    cdef CIndexTensor lt
    cdef int type
    @staticmethod
    cdef wrap_ctensor(CTensor t)
    @staticmethod
    cdef wrap_cindextensor(CIndexTensor t)
    cpdef as_numpy(self)
    cpdef argmax(self, unsigned dim=?, unsigned num=?)
    cpdef categorical_sample_log_prob(self, unsigned dim=?, unsigned num=?)
    
## builders

cdef class RNNState:
    cdef _RNNBuilder builder
    cdef int state_idx
    cdef RNNState _prev
    cdef Expression _out
    cpdef RNNState set_h(self, es=?)
    cpdef RNNState set_s(self, es=?)
    cpdef RNNState add_input(self, Expression x)
    cpdef transduce(self, xs)
    cpdef Expression output(self)
    cpdef tuple h(self)
    cpdef tuple h(self)
    cpdef RNNState prev(self)
    cpdef tuple s(self)

cdef class _RNNBuilder:
    cdef CRNNBuilder *thisptr
    cdef RNNState _init_state
    cdef int cg_version
    cpdef set_dropout(self, float f)
    cpdef disable_dropout(self)
    cdef new_graph(self, update=?)
    cdef Expression add_input(self, Expression e)
    cdef Expression add_input_to_prev(self, CRNNPointer prev, Expression e)
    cdef set_h(self, CRNNPointer prev, es=?)
    cdef set_s(self, CRNNPointer prev, es=?)
    cdef rewind_one_step(self)
    cdef final_h(self)
    cdef final_s(self)
    cdef get_h(self, CRNNPointer i)
    cdef get_s(self, CRNNPointer i)
    cpdef RNNState initial_state(self,vecs=?,update=?)
    cpdef RNNState initial_state_from_raw_vectors(self,vecs=?, update=?)
    cpdef ParameterCollection param_collection(self)
    cdef start_new_sequence(self, es=?)
    cdef Expression back(self)


cdef class LSTMBuilder(_RNNBuilder):
    cdef CLSTMBuilder* thislstmptr
    cdef tuple _spec
    cpdef get_parameters(self)
    cpdef get_parameter_expressions(self)


## trainer

cdef class Trainer:
    cdef CTrainer *thisptr
    cpdef update(self, float s=?)
    cpdef update_epoch(self, float r =?)
    cpdef status(self)
    cpdef set_sparse_updates(self,bool su)
    cpdef set_clip_threshold(self,float thr)
    cpdef get_clip_threshold(self)
    cpdef update_subset(self, updated_params, updated_lookups, float s=?)


cdef class SimpleSGDTrainer(Trainer):
    pass 
    
cdef class CyclicalSGDTrainer(Trainer):
    cdef CCyclicalSGDTrainer *thischildptr

## need to retype the thisptr?

cdef class MomentumSGDTrainer(Trainer):
    pass

cdef class AdagradTrainer(Trainer):
    pass 

cdef class RMSPropTrainer(Trainer):
    pass 

cdef class AdamTrainer(Trainer):
    pass 

cdef ComputationGraph get_cg()

## expression related functions
cpdef Expression softmax(Expression x)
cpdef Expression log(Expression x)
cpdef Expression esum(list xs)
cpdef Expression tanh(Expression x)
cpdef Expression concatenate(list xs, unsigned d=?)
cpdef Expression transpose(Expression x, list dims=?)
cpdef Expression concatenate_cols(list xs)
cpdef Expression colwise_add(Expression x, Expression y)
cpdef Expression select_cols(Expression x, vector[unsigned] cs)
cpdef Expression select_rows(Expression x, vector[unsigned] rs)
cpdef Expression dropout(Expression x, float p)
cdef Expression makeTensor(arr)
cpdef Expression inputTensor(arr,batched=?)
cpdef Expression inputVector(vector[float] v)
cpdef Expression logistic(Expression x)
cpdef Expression cmult(Expression x, Expression y)
cpdef Expression sparsemax(Expression x)
