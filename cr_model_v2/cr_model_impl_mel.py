import tensorflow as tf
from cr_model_v2 import cr_model

class MelModel(cr_model.CGRUFCModel):
    def cnn(self, inputs, seq_lens):
        print('Model')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, False, False, False, False,False, False]
        kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3,3], [3,3], [3,3]]
        filter_nums = [32, 64, 128,128,128,128,128]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],[1,1],[1,1]]
        is_seps = [False, True, True, True, True, True, True,True]
        for ker_size, filter_num, stride, is_pool, is_sep in zip(kernel_sizes, filter_nums, strides,
                                                                 is_poolings, is_seps):
            i += 1
            if is_sep:
                with tf.name_scope('dwise{}'.format(i)):
                    if i == 2:
                        h, seq_lens = self.dwise_no_add(inputs=h, kernel_size=ker_size, filter_num=filter_num,
                                                           seq_lens=seq_lens, strides=stride,
                                                           padding='SAME',
                                                           activation_fn=tf.nn.relu,
                                                           name='dwise{}'.format(i))
                    else:
                        h, seq_lens, = self.dwise_with_add(inputs=h, kernel_size=ker_size, filter_num=filter_num,
                                                             seq_lens=seq_lens, strides=stride,
                                                             padding='SAME',
                                                             activation_fn=tf.nn.relu,
                                                             name='dwise{}'.format(i))

                if is_pool:
                    h, seq_lens = self.maxpooling_seq(inputs=h, pool_size=[3, 3], strides=[2, 2],
                                                         seq_length=seq_lens)

            else:
                with tf.name_scope('conv{}'.format(i)):
                    h, seq_lens = self.conv2d(inputs=h, filters=filter_num, kernel_size=ker_size,
                                                  seq_length=seq_lens, strides=stride, padding='valid',
                                                  use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                                  is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                                  is_training=self.is_training_ph)
                    if is_pool:
                        h, seq_lens = self.maxpooling_seq(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                             seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens

    def fc(self, inputs):
        out_dim = len(self.hps.emos)
        fc_hidden = 64
        inputs = tf.nn.dropout(inputs, self.fc_kprob_ph)
        with tf.name_scope('fc1'):
            h_fc1_act = tf.layers.dense(inputs=inputs, units=fc_hidden,
                                        activation=tf.keras.layers.PReLU())
        with tf.name_scope('fc2'):
            h_fc2 = tf.layers.dense(inputs=h_fc1_act, units=out_dim, activation=None)
        h_fc = h_fc2
        hid_fc = h_fc1_act
        return h_fc, hid_fc

    def dwise_with_add(self, inputs,
                         kernel_size,
                         filter_num,
                         seq_lens,
                         strides,
                         padding,
                         is_add=True,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(0.1),
                         dilation_rate=(1, 1),
                         activation_fn=None,
                         name=None):
        if padding.lower() == 'valid':
            k = (kernel_size[0] - 1) * dilation_rate[0] + 1
            seq_lens = seq_lens - k + 1
        new_seq_len = 1 + tf.floor_div((seq_lens - 1), strides[0])
        bottlen_channel = inputs.get_shape().as_list()[-1]

        output1 = tf.layers.conv2d(inputs, filter_num, kernel_size=[1, 1], padding='same',
                                   activation=activation_fn, kernel_initializer=kernel_initializer)

        with tf.variable_scope(name):
            dwise_filter = tf.get_variable(name='w_f', shape=[kernel_size[0], kernel_size[1], filter_num, 1],
                                           dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer(0.1))

            output2 = tf.nn.depthwise_conv2d(input=output1, filter=dwise_filter,
                                             strides=[1, strides[0], strides[1], 1], padding=padding)

        output2 = activation_fn(output2)

        output3 = tf.layers.conv2d(output2, bottlen_channel, kernel_size=[1, 1], padding='same',
                                   activation=None, kernel_initializer=kernel_initializer)
        if is_add:
            return tf.add(inputs, output3), new_seq_len
        else:
            return output3, new_seq_len

    def dwise_no_add(self, inputs,
                 kernel_size,
                 filter_num,
                 seq_lens,
                 strides,
                 padding,
                 kernel_initializer=tf.contrib.layers.xavier_initializer(0.1),
                 dilation_rate=(1, 1),
                 activation_fn=None,
                 name=None):
        if padding.lower() == 'valid':
            k = (kernel_size[0] - 1) * dilation_rate[0] + 1
            seq_lens = seq_lens - k + 1
        new_seq_len = 1 + tf.floor_div((seq_lens - 1), strides[0])

        output1 = tf.layers.conv2d(inputs, filter_num, kernel_size=[1, 1], padding='same',
                                   activation=activation_fn, kernel_initializer=kernel_initializer)

        with tf.variable_scope(name):
            dwise_filter = tf.get_variable(name='w_f', shape=[kernel_size[0], kernel_size[1], filter_num, 1],
                                           dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer(0.1))

            output2 = tf.nn.depthwise_conv2d(input=output1, filter=dwise_filter,
                                             strides=[1, strides[0], strides[1], 1], padding=padding)

        output2 = activation_fn(output2)

        output3 = tf.layers.conv2d(output2, filter_num, kernel_size=[1, 1], padding='same',
                                   activation=None, kernel_initializer=kernel_initializer)

        return output3, new_seq_len

    def maxpooling_seq(self,inputs,
                      pool_size,
                      strides,
                      seq_length,
                      padding='same',
                      is_seq_mask=False,
                      name=None):
        h = tf.layers.max_pooling2d(inputs=inputs,
                                    pool_size=pool_size,
                                    strides=strides,
                                    padding=padding,
                                    name=name)
        if padding.lower() == 'valid':
            seq_length = seq_length - pool_size[0] + 1
        new_seq_len = 1 + tf.floordiv((seq_length - 1), strides[0])
        if is_seq_mask:
            mask = self.get_mask_4d(new_seq_len, tf.shape(h)[1])
            h = h * mask
        return h, new_seq_len

    def get_mask_4d(self, seq_lens, max_len, dtype=tf.float32):
        """Mask for CNN hidden. [batch_size, max_time, dim(freq), channel]"""
        mask = tf.cast(tf.sequence_mask(seq_lens, max_len), dtype=dtype)
        mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
        return mask

    def conv2d(self, inputs,
               filters,
               kernel_size,
               seq_length,
               is_seq_mask=True,
               is_bn=False,
               is_training=True,
               strides=(1, 1),
               padding='valid',
               dilation_rate=(1, 1),
               activation_fn=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               reuse=None):
        if is_bn:
            use_bias = False
        outputs, seq_len = self.conv2d_with_seq(inputs=inputs,
                                                filters=filters,
                                                kernel_size=kernel_size,
                                                seq_length=seq_length,
                                                strides=strides,
                                                padding=padding,
                                                dilation_rate=dilation_rate,
                                                use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                activity_regularizer=activity_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                bias_constraint=bias_constraint,
                                                trainable=trainable,
                                                name=name,
                                                reuse=reuse)
        if is_bn:
            outputs = tf.contrib.layers.batch_norm(inputs=outputs,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   fused=True,
                                                   reuse=reuse)
        if is_seq_mask:
            mask = self.get_mask_4d(seq_len, tf.shape(outputs)[1], outputs.dtype)
            outputs = outputs * mask  # dot product
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs, seq_len

    def conv2d_with_seq(self, inputs,
                             filters,
                             kernel_size,
                             seq_length,
                             strides,
                             padding,
                             dilation_rate,
                             use_bias=True,
                             kernel_initializer=None,
                             bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=None,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             bias_constraint=None,
                             trainable=True,
                             name=None,
                             reuse=None):
        """inputs, 4d tensor [batch_size, max_time, dim(freq), channel]"""
        if padding.lower() == 'valid':
            k = (kernel_size[0] - 1) * dilation_rate[0] + 1
            seq_length = seq_length - k + 1
        new_seq_len = 1 + tf.floor_div((seq_length - 1), strides[0])

        outputs = tf.layers.conv2d(inputs=inputs,
                                   filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=padding,
                                   data_format='channels_last',
                                   dilation_rate=dilation_rate,
                                   activation=None,
                                   use_bias=use_bias,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   activity_regularizer=activity_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   bias_constraint=bias_constraint,
                                   trainable=trainable,
                                   name=name,
                                   reuse=reuse)

        return outputs, new_seq_len

