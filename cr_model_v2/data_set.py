import collections

import tensorflow as tf


class BatchedInput(
    collections.namedtuple('BatchedInput', ('x', 'e', 't','w','l'))
):
    pass

class BatchedIter(
    collections.namedtuple('BatchedIter', ('initializer', 'BatchedInput'))
):
    pass


class DataSet(object):

    def __init__(self, l_data, hps):
        train_x = tf.data.Dataset.from_generator(lambda: l_data.train_x, tf.float32)
        train_e = tf.data.Dataset.from_tensor_slices(l_data.train_e)
        train_t = tf.data.Dataset.from_tensor_slices(l_data.train_t)
        train_w = tf.data.Dataset.from_tensor_slices(l_data.train_w)
        train_l = tf.data.Dataset.from_tensor_slices(l_data.train_l)
        train_set = tf.data.Dataset.zip((train_x, train_e, train_t, train_w, train_l))
        self.train_set_no_repeat = train_set.padded_batch(hps.infer_batch_size,
                                                          padded_shapes=([None, None], [],[], [],[]))
        train_set = train_set.repeat().shuffle(5000)
        self.train_set = train_set.padded_batch(hps.batch_size,
                                                padded_shapes=([None, None], [], [],[],[]))
        dev_x = tf.data.Dataset.from_generator(lambda: l_data.dev_x, tf.float32)
        dev_e = tf.data.Dataset.from_tensor_slices(l_data.dev_e)
        dev_t = tf.data.Dataset.from_tensor_slices(l_data.dev_t)
        dev_w = tf.data.Dataset.from_tensor_slices(l_data.dev_w)
        dev_l = tf.data.Dataset.from_tensor_slices(l_data.dev_l)
        dev_set = tf.data.Dataset.zip((dev_x, dev_e, dev_t, dev_w, dev_l))
        if hps.is_shuffle_vali:
            dev_set = dev_set.shuffle(5000)
        self.dev_set = dev_set.padded_batch(hps.infer_batch_size,
                                            padded_shapes=([None, None], [], [],[],[]))
        test_x = tf.data.Dataset.from_generator(lambda: l_data.test_x, tf.float32)
        test_e = tf.data.Dataset.from_tensor_slices(l_data.test_e)
        test_t = tf.data.Dataset.from_tensor_slices(l_data.test_t)
        test_w = tf.data.Dataset.from_tensor_slices(l_data.test_w)
        test_l = tf.data.Dataset.from_tensor_slices(l_data.test_l)
        test_set = tf.data.Dataset.zip((test_x, test_e, test_t, test_w, test_l))
        if hps.is_shuffle_test:
            test_set = test_set.shuffle(5000)
        self.test_set = test_set.padded_batch(hps.infer_batch_size,
                                              padded_shapes=([None, None], [], [],[],[]))

    def get_train_iter(self):
        batched_iter = self.train_set.make_initializable_iterator()
        x, e, t,w,l= batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t,
            w=w,
            l=l
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_train_no_repeat_iter(self):
        batched_iter = self.train_set_no_repeat.make_initializable_iterator()
        x, e, t,w, l= batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t,
            w=w,
            l=l
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_dev_iter(self):
        batched_iter = self.dev_set.make_initializable_iterator()
        x, e, t,w, l = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t,
            w=w,
            l=l
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_test_iter(self):
        batched_iter = self.test_set.make_initializable_iterator()
        x, e, t,w, l= batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t,
            w=w,
            l=l
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )
