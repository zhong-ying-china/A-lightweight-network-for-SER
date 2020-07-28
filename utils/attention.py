import tensorflow as tf


def fc_mask(seq_lens, max_len):
    mask = tf.cast(tf.sequence_mask(seq_lens, max_len), dtype=tf.float32)
    mask = tf.expand_dims(mask, 2)
    return mask


def attention(inputs, attention_hidden_size, seq_lens, time_major=False):
    """
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_hidden_size: Linear size of the Attention weights.
        seq_lens: [batch_size], available sequence length
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.

    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w1 = tf.Variable(tf.random_normal([hidden_size, attention_hidden_size], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([attention_hidden_size], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([attention_hidden_size, 1], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([1], stddev=0.1))
    mask = fc_mask(seq_lens, tf.shape(inputs)[1])

    with tf.name_scope('attention1'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_hidden_size

        h1 = tf.tanh((tf.tensordot(inputs, w1, axes=1) + b1) * mask)

    with tf.name_scope('attention2'):
        # (B, T, A) => (B, T, 1)
        h2 = (tf.tensordot(h1, w2, axes=1) + b2) * mask

    # softmax implement

    numerator = tf.exp(h2) * mask

    denominator = tf.reduce_sum(numerator, axis=1, keepdims=True)

    alphas = tf.divide(numerator, denominator)
    # print(tf.multiply(inputs, alphas).shape)
    outputs = tf.reduce_sum(tf.multiply(inputs, alphas), axis=1, keepdims=False)
    # print(outputs.shape)
    # outputs = outputs.reshape()
    # outputs = tf.multiply(inputs, alphas)

    return outputs, alphas
