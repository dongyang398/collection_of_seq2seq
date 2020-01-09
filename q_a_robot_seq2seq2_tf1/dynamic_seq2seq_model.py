# -*- coding:utf-8 -*-
import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMStateTuple


class dynamicSeq2seq():
    '''
    Dynamic_Rnn_Seq2seq with Tensorflow-1.0.0

        args:
        encoder_cell            encoder结构
        decoder_cell            decoder结构
        encoder_vocab_size      encoder词典大小
        decoder_vocab_size      decoder词典大小
        embedding_size          embedd成的维度
        bidirectional           encoder的结构
                                True:  encoder为双向LSTM
                                False: encoder为一般LSTM
        attention               decoder的结构
                                True:  使用attention模型
                                False: 一般seq2seq模型
        time_major              控制输入数据格式
                                True:  [time_steps, batch_size]
                                False: [batch_size, time_steps]


    '''
    PAD = 0
    EOS = 2
    UNK = 3

    def __init__(self, encoder_cell,
                 decoder_cell,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 embedding_size,
                 bidirectional=True,
                 attention=False,
                 debug=False,
                 time_major=False):

        self.hidden_dim = 50
        self.keep_prob = 0.5
        self.rnn_layers =2
        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size

        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.global_step = tf.Variable(-1, trainable=False)
        self.max_gradient_norm = 5
        self.time_major = time_major

        # 创建模型
        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _make_graph(self):
        # 创建占位符
        self._init_placeholders()

        # 兼容decoder输出数据
        self._init_decoder_train_connectors()

        # embedding层
        self._init_embeddings()

        # 创建encoder
        self._init_encoder()

        # 创建decoder，会判断是否使用attention模型
        self._init_decoder()

        # 计算loss及优化
        self._init_optimizer()

    # 初始化占位符
    def _init_placeholders(self):

        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )

        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(
                tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat(
                [EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1
            # self.decoder_train_length = self.decoder_targets_length

            decoder_train_targets = tf.concat(
                [self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(
                tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(
                decoder_train_targets_eos_mask, [1, 0])

            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    # 嵌入降维
    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.encoder_embedding_matrix = tf.get_variable(
                name="encoder_embedding_matrix",
                shape=[self.encoder_vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.decoder_embedding_matrix = tf.get_variable(
                name="decoder_embedding_matrix",
                shape=[self.decoder_vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            # encoder的embedd
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.encoder_embedding_matrix, self.encoder_inputs)

            # decoder的embedd
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.decoder_embedding_matrix, self.decoder_train_inputs)

    def build_rnn_cell(self):
        def single_rnn_cell():
            cell = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell

        rnn_cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.rnn_layers)])
        return rnn_cell

    def _init_encoder(self):
        '''
        encdoer
        '''
        with tf.variable_scope("Encoder") as scope:
            encoder_lstm_cell = self.build_rnn_cell()
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(encoder_lstm_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=self.time_major,
                                  dtype=tf.float32)
            )


    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                self.test_outputs = outputs
                return tf.contrib.layers.linear(outputs, self.decoder_vocab_size, scope=scope)
        with tf.variable_scope("attention"):
            decoder_lstm_cell = self.build_rnn_cell()
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.hidden_dim, self.encoder_outputs)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_lstm_cell, attention_mechanism, self.hidden_dim)
            decoder_initial_state = decoder_cell.zero_state(tf.shape(self.encoder_inputs)[0], dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=self.encoder_state)

            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(
                    encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.decoder_embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(
                        self.encoder_inputs_length) + 100,
                    num_decoder_symbols=self.decoder_vocab_size,
                )
            else:

                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(
                    self.encoder_outputs, [1, 0, 2])

                (attention_keys,
                 attention_values,
                 attention_score_fn,
                 attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.decoder_embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(
                        self.encoder_inputs_length) + 100,
                    num_decoder_symbols=self.decoder_vocab_size,
                )

            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=self.time_major,
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(
                self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=self.time_major,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(
                self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    def _init_MMI(self, logits, targets):
        sum_mmi = 0
        x_value_list = 1

    # 整理输出并计算loss
    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        self.targets = tf.transpose(self.decoder_train_targets, [1, 0])

        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)

        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss)

        # add
        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []

        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())
