import tensorflow as tf


def get_mask(seq_lens, max_len, dtype=tf.float32):
    mask = tf.cast(tf.sequence_mask(seq_lens, max_len), dtype=dtype)
    mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
    return mask


def get_mask_3d(seq_lens, max_len, dtype=tf.float32):
    mask = tf.cast(tf.sequence_mask(seq_lens, max_len), dtype=dtype)
    mask = tf.expand_dims(mask, -1)
    return mask


def var_conv2d(inputs, w, strides, padding, bias, seq_length, is_clip_output_size=False):
    """
    conv2d process variable length sequence input, without activate function.
    :param inputs: A tensor, [batch_size, max_time, dim, channel].
    :param w: A tensor, [filter_height, filter_width, in_channels, out_channels].
    :param strides: A list of ints. 1-D tensor of length 4.
    :param padding: A string from: "SAME", "VALID".
    :param bias: bias for cnn.
    :param seq_length: A tensor, [batch_size].
    :param is_clip_output_size: clip useless padding value in output
    :return: output, new_seq_len
    """
    h = tf.nn.conv2d(input=inputs, filter=w, strides=strides, padding=padding) + bias
    s = strides[1]
    seq_len1 = seq_length
    if padding == 'VALID':
        k = tf.shape(w)[0]
        seq_len1 = seq_length - k + 1
    new_seq_len = 1 + tf.floordiv((seq_len1 - 1), s)
    mask = get_mask(new_seq_len, tf.shape(h)[1], h.dtype)
    outputs = h * mask
    if is_clip_output_size:
        max_seq_len = tf.reduce_max(new_seq_len)
        outputs = outputs[:, :max_seq_len, :, :]
    return outputs, new_seq_len


# todo: test
def var_conv2d_v2(inputs, w, bias, seq_length, strides=(1, 2, 2, 1), padding='SAME',
                  is_training=True,
                  activation_fn=tf.nn.relu, is_bn=False, is_mask=True, reuse=None):
    if is_bn:
        h = tf.nn.conv2d(input=inputs, filter=w, strides=strides, padding=padding)
    else:
        h = tf.nn.conv2d(input=inputs, filter=w, strides=strides, padding=padding) + bias
    s = strides[1]
    seq_len1 = seq_length
    if padding == 'VALID':
        k = tf.shape(w)[0]
        seq_len1 = seq_length - k + 1
    new_seq_len = 1 + tf.floordiv((seq_len1 - 1), s)
    if is_bn:
        h = tf.contrib.layers.batch_norm(inputs=h,
                                         center=True,
                                         scale=True,
                                         updates_collections=None,
                                         is_training=is_training,
                                         fused=True,
                                         reuse=reuse)
    if is_mask:
        mask = get_mask(new_seq_len, tf.shape(h)[1], h.dtype)
        outputs = h * mask
    else:
        outputs = h
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs, new_seq_len


# todo: test
def var_conv1d(inputs, w, bias, seq_length, stride=1, padding='SAME',
               is_training=True,
               activation_fn=tf.nn.relu, is_bn=False, is_mask=True, reuse=None):
    if is_bn:
        h = tf.nn.conv1d(value=inputs, filters=w, stride=stride, padding=padding)
        h = tf.contrib.layers.batch_norm(inputs=h,
                                         center=True,
                                         scale=True,
                                         updates_collections=None,
                                         is_training=is_training,
                                         fused=True,
                                         reuse=reuse)
    else:
        h = tf.nn.conv1d(value=inputs, filters=w, stride=stride, padding=padding) + bias
    s = stride
    seq_len1 = seq_length
    if padding == 'VALID':
        k = tf.shape(w)[0]
        seq_len1 = seq_length - k + 1
    new_seq_len = 1 + tf.floordiv((seq_len1 - 1), s)
    if is_mask:
        mask = get_mask_3d(new_seq_len, tf.shape(h)[1], h.dtype)
        outputs = h * mask
    else:
        outputs = h
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs, new_seq_len


def var_max_pool(inputs, ksize, strides, padding, seq_length, is_clip_output_size=False):
    """
    max pool for variable length sequence input,
    if padding == 'SAME', please make sure all the element in inputs >= 0
    :param inputs: A Tensor, [batch_size, max_time, dim, channel].
    :param ksize: A 1-D int Tensor of 4 elements.
    :param strides: A 1-D int Tensor of 4 elements.
    :param padding: A string, either 'VALID' or 'SAME'.
    :param seq_length: A tensor, [batch_size].
    :param is_clip_output_size: clip useless padding value in output
    :return: output, new_seq_len
    """
    h = tf.nn.max_pool(value=inputs, ksize=ksize, strides=strides, padding=padding)
    s = strides[1]
    seq_len1 = seq_length
    if padding == 'VALID':
        k = ksize[1]
        seq_len1 = seq_length - k + 1
    new_seq_len = 1 + tf.floordiv((seq_len1 - 1), s)
    mask = get_mask(new_seq_len, tf.shape(h)[1], h.dtype)
    outputs = h * mask
    if is_clip_output_size:
        max_seq_len = tf.reduce_max(new_seq_len)
        outputs = outputs[:, :max_seq_len, :, :]
    return outputs, new_seq_len


def var_max_pool_3d(inputs, ksize, strides, padding, seq_length, is_clip_output_size=False):
    """
    max pool for variable length sequence input,
    if padding == 'SAME', please make sure all the element in inputs >= 0
    :param inputs: A Tensor, [batch_size, max_time, channel].
    :param ksize: A 1-D int Tensor of 4 elements.
    :param strides: A 1-D int Tensor of 4 elements.
    :param padding: A string, either 'VALID' or 'SAME'.
    :param seq_length: A tensor, [batch_size].
    :param is_clip_output_size: clip useless padding value in output
    :return: output, new_seq_len
    """
    h = tf.nn.pool(input=inputs, window_shape=ksize, strides=strides, padding=padding,
                   pooling_type='MAX')
    s = strides[0]
    seq_len1 = seq_length
    if padding == 'VALID':
        k = ksize[0]
        seq_len1 = seq_length - k + 1
    new_seq_len = 1 + tf.floordiv((seq_len1 - 1), s)
    mask = get_mask_3d(new_seq_len, tf.shape(h)[1], h.dtype)
    outputs = h * mask
    if is_clip_output_size:
        max_seq_len = tf.reduce_max(new_seq_len)
        outputs = outputs[:, :max_seq_len, :]
    return outputs, new_seq_len


def var_bn(inputs, seq_length=None, is_training=True, activation_fn=None, scope="bn", reuse=None):
    """Applies batch normalization.
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension. Or if type is `ln`, the normalization is over
        the last dimension. Note that this is different from the native
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py`
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      seq_length: A tensor, [batch_size].
      is_training: Whether or not the layer is in training mode.
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
    # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               fused=True,
                                               reuse=reuse)
        if seq_length is not None:
            mask = get_mask(seq_length, tf.shape(inputs)[1], inputs.dtype)
            outputs = outputs * mask
        # restore original shape
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)
    else:  # fallback to naive batch norm
        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               reuse=reuse,
                                               fused=False)
    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs
def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    v = tf.sigmoid(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)  # (B,T) shape
    alphas = tf.nn.softmax(vu)  # (B,T) shape also

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas