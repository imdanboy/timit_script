import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.layers import convolutional as layers_conv
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
import numpy as np
import sys, os, time, argparse

phn_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
phn_39 = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', 'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', 'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#', 'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

train_data_path = 'data/fbank/train.tfrecords'
dev_data_path = 'data/fbank/dev.tfrecords'
test_data_path = 'data/fbank/test.tfrecords'
checkpoints_path = 'model/{}/ckpt'

feat_type = 'fbank'
feats_dim = 39 if feat_type=='mfcc' else 123
labels_sos_id = len(phn_61) + 1
labels_eos_id = len(phn_61) 
num_classes = len(phn_61) + 1

num_encoder = 256
num_decoder = 256
keep_prob = 0.5
n_encoder_layer = 3 
beam_width = 10
batch_size = 32
epochs = 80

def _luong_score(query, keys, scale, conv_feat, score_dense):
    depth = query.get_shape()[-1]
    key_units = keys.get_shape()[-1]
    if depth != key_units:
        raise ValueError()
    query = tf.expand_dims(query, 1) # [batch, 1, num_decoder]
    score = score_dense(tf.tanh(keys + query + conv_feat))
    score = tf.squeeze(score, -1)
    #score = tf.matmul(query, keys, transpose_b=True)
    #score = tf.squeeze(score, [1])
    return score # [batch, max_time]

class MyLuongAttention(tf.contrib.seq2seq.LuongAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="LuongAttention"):
        if probability_fn is None:
            probability_fn = tf.nn.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(tf.contrib.seq2seq.LuongAttention, self).__init__(
            query_layer=None,
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._scale = scale
        self._name = name
        self.query_dense = layers_core.Dense(num_units)
        self.cf_conv = layers_conv.Conv1D(filters=64, kernel_size=201, strides=1, padding='same')
        self.cf_dense = layers_core.Dense(num_units)
        self.score_dense = layers_core.Dense(1)

    def __call__(self, query, state):
        with tf.variable_scope(None, "luong_attention", [query]):
            expanded_state = tf.expand_dims(state, -1)
            conv_feat = self.cf_conv(expanded_state)
            conv_feat = self.cf_dense(conv_feat)
            query = self.query_dense(query)
            score = _luong_score(query, self._keys, self._scale, conv_feat, self.score_dense)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer, attention_dropout_layer, training):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(cell_output, state=attention_state)

    # attention_state = [batch, max_time], (previous alignments)
    # alignments = next_attention_state = [batch, max_time]
    expanded_alignments = tf.expand_dims(alignments, 1)
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1]) # [batch, 2*num_encoder]

    if attention_layer is not None:
        raise NotImplementedError()
    else:
        attention = context

    return attention, alignments, next_attention_state    
    
class MyAttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    
    def __init__(self, cell, attention_mechanism, keep_prob, training, attention_layer_size=None, alignment_history=False, cell_input_fn=None,
                output_attention=True, initial_cell_state=None, name=None):
        super(MyAttentionWrapper, self).__init__(cell, attention_mechanism, attention_layer_size, alignment_history, cell_input_fn,
                output_attention, initial_cell_state, name)
        
        self.keep_prob = keep_prob
        self.training = training
        
        super(tf.contrib.seq2seq.AttentionWrapper, self).__init__(name=name)
        if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
            raise TypeError("cell must be an RNNCell, saw type: %s" % type(cell).__name__)
        if isinstance(attention_mechanism, (list, tuple)):
            self._is_multi = True
            attention_mechanisms = attention_mechanism
            for attention_mechanism in attention_mechanisms:
                if not isinstance(attention_mechanism, AttentionMechanism):
                    raise TypeError("attention_mechanism must contain only instances of "
                                    "AttentionMechanism, saw type: %s" % type(attention_mechanism).__name__)
        else:
            self._is_multi = False
            if not isinstance(attention_mechanism, tf.contrib.seq2seq.AttentionMechanism):
                raise TypeError("attention_mechanism must be an AttentionMechanism or list of "
                                "multiple AttentionMechanism instances, saw type: %s" % type(attention_mechanism).__name__)
            attention_mechanisms = (attention_mechanism,)

        if cell_input_fn is None:
            cell_input_fn = (lambda inputs, attention: tf.concat([inputs, attention], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError("cell_input_fn must be callable, saw type: %s" % type(cell_input_fn).__name__)

        if attention_layer_size is not None:
            attention_layer_sizes = tuple(attention_layer_size if isinstance(attention_layer_size, (list, tuple))
                                              else (attention_layer_size,))
            if len(attention_layer_sizes) != len(attention_mechanisms):
                raise ValueError("If provided, attention_layer_size must contain exactly one "
                                "integer per attention_mechanism, saw: %d vs %d" % (len(attention_layer_sizes), len(attention_mechanisms)))
            self._attention_layers = tuple(layers_core.Dense(attention_layer_size, name="attention_layer", use_bias=True,
                    activation=tf.tanh) for attention_layer_size in attention_layer_sizes)
            self._attention_dropout_layers = tuple(layers_core.Dropout(rate=1-self.keep_prob, name="attention_dropout_layer")
                                                  for attention_layer_size in attention_layer_sizes)
            self._attention_layer_size = sum(attention_layer_sizes)
        else:
            self._attention_layers = None
            self._attention_dropout_layers = None
            self._attention_layer_size = sum(attention_mechanism.values.get_shape()[-1].value
                                                      for attention_mechanism in attention_mechanisms)
            
        self._cell = cell
        self._attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        with tf.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (final_state_tensor.shape[0].value or tf.shape(final_state_tensor)[0])
                error_message = ('custom error msg:0')
                with tf.control_dependencies(
                    self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(lambda s: tf.identity(s, name="check_initial_cell_state"),
                                                              initial_cell_state)
        self.attention_layer = layers_core.Dense(512, activation=tf.tanh)
                    
    def call(self, inputs, state):

        if not isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
            raise TypeError("Expected state to be instance of MyAttentionWrapperState. "
                          "Received type %s instead."  % type(state))
        
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = ('custom error msg:1')
        with tf.control_dependencies(
            self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
              cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = _compute_attention(
            attention_mechanism, cell_output, previous_attention_state[i],
            self._attention_layers[i] if self._attention_layers else None,
            self._attention_dropout_layers[i] if self._attention_dropout_layers else None,
            self.training)
            alignment_history = previous_alignment_history[i].write(
            state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)
        
        next_state = tf.contrib.seq2seq.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        outputs = layers_core.Dropout(rate=1-keep_prob)(self.attention_layer(tf.concat([cell_output, attention], 1)), training=self.training)
        
        if self._output_attention:
            return outputs, next_state
        else:
            raise NotImplementedError()
            #return cell_output, next_state

def input_fn(data_path):
    dataset = tf.data.TFRecordDataset(data_path)
    
    context_features = {'feats_seq_len': tf.FixedLenFeature([], dtype=tf.int64),
                           'labels_seq_len': tf.FixedLenFeature([], dtype=tf.int64)}
    sequence_features = {'features': tf.FixedLenSequenceFeature([feats_dim], dtype=tf.float32),
                        'labels': tf.FixedLenSequenceFeature([], dtype=tf.int64)}
    dataset = dataset.map(lambda serialized_example: tf.parse_single_sequence_example(serialized_example,
                                                                                     context_features=context_features,
                                                                                     sequence_features=sequence_features))
    dataset = dataset.map(lambda context, sequence: (sequence['features'], sequence['labels'], 
                                                     context['feats_seq_len'], context['labels_seq_len']))
    dataset = dataset.map(lambda features, labels, feats_seq_len, labels_seq_len: (features, 
                                                    tf.concat(([labels_sos_id], labels),0),
                                                    tf.concat((labels, [labels_eos_id]), 0),
                                                            feats_seq_len, labels_seq_len))
    dataset = dataset.map(lambda features, labels_in, labels_out, feats_seq_len, labels_seq_len: 
                                      (features, labels_in, labels_out, feats_seq_len, tf.size(labels_in, out_type=tf.int64)))
    def batching_func(x):
        return x.padded_batch(batch_size,
                                 padded_shapes=(tf.TensorShape([None, feats_dim]),
                                               tf.TensorShape([None]),
                                               tf.TensorShape([None]),
                                               tf.TensorShape([]),
                                               tf.TensorShape([])),
                                 padding_values=(tf.cast(0, tf.float32),
                                                tf.cast(labels_eos_id, tf.int64),
                                                tf.cast(labels_eos_id, tf.int64),
                                                tf.cast(0, tf.int64),
                                                tf.cast(0, tf.int64)))
        
    def key_func(features, labels_in, labels_out, feats_seq_len, labels_in_seq_len):
        f0 = lambda: tf.constant(0, tf.int64)
        f1 = lambda: tf.constant(1, tf.int64)
        f2 = lambda: tf.constant(2, tf.int64)
        f3 = lambda: tf.constant(3, tf.int64)
        f4 = lambda: tf.constant(4, tf.int64)
        f5 = lambda: tf.constant(5, tf.int64)
        f6 = lambda: tf.constant(6, tf.int64)
            
        return tf.case([(tf.less_equal(feats_seq_len, 200), f0),
                   (tf.less_equal(feats_seq_len, 250), f1),
                   (tf.less_equal(feats_seq_len, 300), f2),
                   (tf.less_equal(feats_seq_len, 350), f3),
                   (tf.less_equal(feats_seq_len, 400), f4),
                   (tf.less_equal(feats_seq_len, 500), f5)], default=f6)
        
    def reduce_func(bucket_id, windowed_data):
        return batching_func(windowed_data)
    
    if data_path == train_data_path:
        dataset = dataset.shuffle(4000)
        '''
        # tf issue: https://github.com/tensorflow/tensorflow/issues/17932
        batched_dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                                            element_length_func=lambda args_dic: args_dic['feats_seq_len'],
                                            bucket_boundaries=(200,250,300,350,400,500),
                                            bucket_batch_sizes=[batch_size]*7))
        '''
        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
        batched_dataset = batched_dataset.shuffle(150)
    else: # eval or infer, no shuffle
        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
        
    batched_dataset = batched_dataset.prefetch(1)
    return batched_dataset.make_initializable_iterator()


def model_fn(features, labels, mode, params):
    # mode: 'train', 'dev', 'test'
    
    def compute_encoder_outputs():
        def residual_block(inp, out_channels):
            inp_channels = inp.get_shape().as_list()[1]
            out = tf.layers.conv2d(inp, filters=out_channels, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, data_format='channels_first')
            out = tf.layers.batch_normalization(out, axis=1, training=training, scale=False) 
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, rate=1-keep_prob, training=training)
            out = tf.layers.conv2d(out, filters=out_channels, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, data_format='channels_first')
            out = tf.layers.batch_normalization(out, axis=1, training=training, scale=False)
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, rate=1-keep_prob, training=training)
            if inp_channels != out_channels:
                inp = tf.layers.conv2d(inp, filters=out_channels, kernel_size=(1,1), strides=(1,1), padding='same', data_format='channels_first')
            return out + inp
        def conv_block(inp, filters, kernel_size, strides):
            out = tf.layers.conv2d(inp, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, data_format='channels_first')
            out = tf.layers.batch_normalization(out, axis=1, training=training, scale=False) 
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, rate=1-keep_prob, training=training)
            return out
        def dense_block(inp, num_units):
            out = tf.layers.dense(inp, num_units, use_bias=False)
            out = tf.layers.batch_normalization(out, axis=-1, training=training, scale=False) 
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, rate=1-keep_prob, training=training)
            return out
        
        inputs = tf.stack(tf.split(features, num_or_size_splits=3, axis=2), axis=3)
        inputs = tf.transpose(inputs, [0,3,2,1]) # shape = [batch, channels, feats_dim/3, max_time] := NCHW
        
        conv = conv_block(inputs, 256, (3,3), (1,3))
        conv = residual_block(conv, 128)
        conv = residual_block(conv, 128)
        conv = residual_block(conv, 128)
        
        flattend = tf.transpose(conv, [3,0,1,2]) # [time, batch, channels, feats_dim/3]
        flattend = tf.reshape(flattend, [tf.shape(flattend)[0], tf.shape(flattend)[1], 41*128]) # time major
        
        fc = dense_block(flattend, 1024)
        inputs = fc
        
        lstm_layer = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=n_encoder_layer, num_units=num_encoder, direction='bidirectional', dropout=1-keep_prob, kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        lstm_layer.build(inputs.shape)
        outputs, _ = lstm_layer(inputs, training=training)
        outputs = tf.transpose(outputs, [1,0,2]) # to batch major
        
        outputs = tf.layers.dropout(outputs, rate=1-keep_prob, training=training)
        
        memory = outputs
        mem_seq_len = feats_seq_len // 3
        return (memory, mem_seq_len)
    
    def get_decoder_cell_and_init_state():    
        num_batch = tf.shape(memory)[0]
        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_decoder),
                                                     output_keep_prob=keep_prob)
        #decoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_decoder, use_peepholes=False, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)),
        #                                             output_keep_prob=keep_prob)

        memory_tiled = memory
        mem_seq_len_tiled = mem_seq_len
        if mode == 'test':
            memory_tiled = tf.contrib.seq2seq.tile_batch(memory_tiled, multiplier=beam_width)
            mem_seq_len_tiled = tf.contrib.seq2seq.tile_batch(mem_seq_len_tiled, multiplier=beam_width)
            num_batch = num_batch * beam_width

        align_hist = False if mode=='test' else True
        attention_mechanism = MyLuongAttention(num_decoder, memory_tiled, mem_seq_len_tiled, scale=False)
        decoder_cell = MyAttentionWrapper(decoder_cell, attention_mechanism, keep_prob, training,
                                          attention_layer_size=None, alignment_history=align_hist, output_attention=True)
        decoder_initial_state = decoder_cell.zero_state(num_batch, tf.float32)
        return (decoder_cell, decoder_initial_state)
    
    labels_in = labels[0]
    labels_out = labels[1]
    feats_seq_len = tf.to_int32(labels[2])
    labels_in_seq_len = tf.to_int32(labels[3])
    training = (mode == 'train')
    if mode == 'train':
        learning_rate = params['lr']
    
    embedding_decoder = tf.Variable(tf.concat([np.identity(num_classes-1), tf.zeros([2,num_classes-1])], axis=0),
                                        dtype=tf.float32, trainable=False)
    #embedding_decoder = tf.Variable(tf.random_uniform([num_classes+1, 30], minval=-0.05, maxval=0.05, dtype=tf.float32))
    decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, labels_in)
    
    memory, mem_seq_len = compute_encoder_outputs()
    with tf.variable_scope('decoder') as decoder_scope:
        decoder_cell, decoder_initial_state = get_decoder_cell_and_init_state()
    
        output_layer = layers_core.Dense(num_classes, use_bias=True)

        loss, train_op, (per, mapping_table_init_op) = None, None, (None, None)
        
        if mode == 'train':
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, labels_in_seq_len, time_major=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state=decoder_initial_state)
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False, scope=decoder_scope)
            logits = output_layer(outputs.rnn_output) # [batch, max_time, num_classes]
            
            # seq loss calulation method: https://www.tensorflow.org/tutorials/seq2seq#loss
            max_time = tf.shape(labels_out)[1]
            target_weights = tf.sequence_mask(labels_in_seq_len, max_time, dtype=logits.dtype)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_out, logits=logits)
            loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(tf.shape(logits)[0])
            '''
            note: differ from above, check github source code
            loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=labels_out, weights=target_weights,
                                                   average_across_timesteps=False)
            loss = tf.reduce_sum(loss)
            '''
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                params = tf.trainable_variables()
                gradients = tf.gradients(loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                #clipped_gradients = gradients
                train_op = optimizer.apply_gradients(zip(clipped_gradients, params))
            
        elif mode == 'dev':
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, labels_in_seq_len, time_major=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state=decoder_initial_state)
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False, scope=decoder_scope)
            logits = output_layer(outputs.rnn_output)
            
            max_time = tf.shape(labels_out)[1]
            target_weights = tf.sequence_mask(labels_in_seq_len, max_time, dtype=logits.dtype)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_out, logits=logits)
            loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(tf.shape(logits)[0])
            
        elif mode == 'test':
            beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, embedding_decoder, 
                                                tf.fill([tf.shape(features)[0]], labels_sos_id), labels_eos_id,
                                                        decoder_initial_state, beam_width, output_layer=output_layer)
            decoded, _, final_seq_len = tf.contrib.seq2seq.dynamic_decode(beam_decoder, maximum_iterations=100,
                                                                          output_time_major=False, scope=decoder_scope)
            predicted_ids = decoded.predicted_ids # shape = [batch, max_time, beam_width]
            
            def get_sparse_tensor(dense, default):
                indices = tf.to_int64(tf.where(tf.not_equal(dense, default)))
                vals = tf.to_int32(tf.gather_nd(dense, indices))
                shape = tf.to_int64(tf.shape(dense))
                return tf.SparseTensor(indices, vals, shape)
    
            def compute_per():
                '''
                return tuple: (unnormalized edit distance, seqeucne_length),
                it is just sum of batched data
                '''
                phn_61_tensor = tf.constant(phn_61, dtype=tf.string)
                phn_39_tensor = tf.constant(phn_39, dtype=tf.string)
                mapping_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(list(mapping.keys()), list(mapping.values())), default_value='')

                def map_to_reduced_phn(p):
                    val = mapping_table.lookup(phn_61_tensor[p])
                    f1 = lambda: tf.to_int32(tf.reduce_min(tf.where(tf.equal(val, phn_39_tensor))))
                    f2 = lambda: tf.to_int32(tf.reduce_min(tf.where(tf.equal(phn_61_tensor[p], phn_39_tensor))))
                    return tf.cond(tf.not_equal(val, ''), f1, f2)

                indices = tf.to_int64(tf.where(tf.logical_and(tf.not_equal(predicted_ids[:,:,0], -1), 
                                                              tf.not_equal(predicted_ids[:,:,0], labels_eos_id))))
                vals = tf.to_int32(tf.gather_nd(predicted_ids[:,:,0], indices))
                shape = tf.to_int64(tf.shape(predicted_ids[:,:,0]))
                decoded_sparse = tf.SparseTensor(indices, vals, shape)
                labels_out_sparse = get_sparse_tensor(labels_out, labels_eos_id)

                decoded_reduced = tf.SparseTensor(decoded_sparse.indices, tf.map_fn(map_to_reduced_phn, decoded_sparse.values), decoded_sparse.dense_shape)
                labels_out_reduced = tf.SparseTensor(labels_out_sparse.indices, tf.map_fn(map_to_reduced_phn, labels_out_sparse.values), labels_out_sparse.dense_shape)

                
                per_tuple = tf.reduce_sum(tf.edit_distance(decoded_reduced, labels_out_reduced, normalize=False)), tf.to_float(tf.size(labels_out_reduced.values))
                return (per_tuple, mapping_table.init)
            per, mapping_table_init_op = compute_per()
            
    return loss, train_op, (per, mapping_table_init_op)


if __name__ == '__main__':

    checkpoints_path = checkpoints_path.format(sys.argv[0].split('.')[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', help='index of gpu', default='0')
    parser.add_argument('-e', dest='ft_epoch', help='epoch from which the finetunning starts', type=int, default=65)
    args = parser.parse_args()

    train_graph = tf.Graph()
    dev_graph = tf.Graph()
    test_graph = tf.Graph()
    
    with train_graph.as_default():
        iter_train = input_fn(train_data_path)
        train_feats_labels = iter_train.get_next()
        lr = tf.placeholder(dtype=tf.float32)
        train_loss, train_op, _ = model_fn(train_feats_labels[0], train_feats_labels[1:], 'train', params={'lr':lr})
        train_saver = tf.train.Saver(max_to_keep=5)
        var_init_train_op = tf.global_variables_initializer()
        
    with dev_graph.as_default():
        iter_dev = input_fn(dev_data_path)
        dev_feats_labels = iter_dev.get_next()
        dev_loss, _, _ = model_fn(dev_feats_labels[0], dev_feats_labels[1:], 'dev', params=None)
        dev_saver = tf.train.Saver()
        
    with test_graph.as_default():
        iter_test = input_fn(test_data_path)
        test_feats_labels = iter_test.get_next()
        
        _, _, (test_per, mapping_table_init_op) = model_fn(test_feats_labels[0], test_feats_labels[1:], 'test', params=None)
        test_saver = tf.train.Saver()
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    #sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess_config.gpu_options.visible_device_list=args.gpu
    
    train_sess = tf.Session(graph=train_graph, config=sess_config)
    train_sess.run(var_init_train_op)
    
    dev_sess = tf.Session(graph=dev_graph, config=sess_config)
    
    test_sess = tf.Session(graph=test_graph, config=sess_config)
    test_sess.run(mapping_table_init_op)

    dev_loss_np = []
    #train_saver.restore(train_sess, checkpoints_path + '-26') 
    #for epoch in range(26,epochs):
    for epoch in range(epochs):
        train_sess.run(iter_train.initializer)
        train_losses = []
        start = time.time()
        while True:
            try:
                feed_dict = {lr: 0.001} if epoch+1<args.ft_epoch else {lr: 0.0001}
                train_sess.run(train_op, feed_dict=feed_dict)
                train_losses.append(train_sess.run(train_loss))
            except tf.errors.OutOfRangeError:
                duration = time.time() - start
                log = 'Epoch {}/{}, \ntrain_loss={:.3f}, time={:.0f}s'
                print(log.format(epoch+1, epochs, np.mean(train_losses), duration))
            
                if not os.path.isdir(os.path.split(checkpoints_path)[0]):
                    os.makedirs(os.path.split(checkpoints_path)[0])
                saved_ckpt_path = train_saver.save(train_sess, checkpoints_path, global_step=epoch+1)
                
                dev_saver.restore(dev_sess, saved_ckpt_path)
                dev_sess.run(iter_dev.initializer)
                dev_losses = []
                start = time.time()
                while True:
                    try:
                        dev_losses.append(dev_sess.run(dev_loss))
                    except tf.errors.OutOfRangeError:
                        duration = time.time() - start
                        log = '\tdev_loss={:.3f}, time={:.0f}s'
                        print(log.format(np.mean(dev_losses), duration))
                        dev_loss_np.append(np.mean(dev_losses))
                        break    
                    
                if epoch+1 >= args.ft_epoch:
                    test_saver.restore(test_sess, saved_ckpt_path)
                    test_sess.run(iter_test.initializer)
                    test_unnormed_edit_dist = []
                    test_seq_len = []
                    start = time.time()
                    while True:
                        try:
                            unnormed_edit_dist, seq_len = test_sess.run(test_per)
                            test_unnormed_edit_dist.append(unnormed_edit_dist)
                            test_seq_len.append(seq_len)
                        except tf.errors.OutOfRangeError:
                            duration = time.time() - start
                            log = '\t\ttest_per={:.3f}, time={:.0f}s'
                            print(log.format(sum(test_unnormed_edit_dist)/sum(test_seq_len), duration))
                            break
                
                break

    np.save(sys.argv[0].split('.')[0], np.array(dev_loss_np))         
    train_sess.close()
    dev_sess.close()
    test_sess.close()

