import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMCell

def conv1d(in_, filter_size, conv2d, height, padding, is_train=True, drop_rate=0.0, scope=None):
    with tf.compat.v1.variable_scope(scope or "conv1d"):
        in_ = tf.keras.layers.Dropout( rate=drop_rate)(in_)
        xxc = conv2d(in_)
        out = tf.math.reduce_max(tf.nn.relu(xxc), axis=2)  # max-pooling, [-1, max_len_sent, char output size]
        return out


def multi_conv1d(in_, filter_sizes,heights, padding="VALID", is_train=True, drop_rate=0.0, scope=None,conv2d = None):

   
    assert len(filter_sizes) == len(heights)
    outs = []
    for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
        if filter_size == 0:
            continue
        out = conv1d(in_, filter_size, conv2d, height, padding, is_train=is_train, drop_rate=drop_rate,
                        scope="conv1d_{}".format(i))
        outs.append(out)
    concat_out = tf.concat(axis=2, values=outs)

    return concat_out


class AttentionCell(RNNCell):
    def __init__(self, num_units, memory, pmemory, cell_type='lstm'):
        super(AttentionCell, self).__init__()
        self._cell = LSTMCell(num_units)
        self.num_units = num_units
        self.memory = memory
        self.pmemory = pmemory
        self.mem_units = memory.get_shape().as_list()[-1]

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        # Attention model
        c, m = state # c is previous cell state, m is previous hidden state
        ha = tf.nn.tanh(tf.add(self.pmemory, tf.keras.layers.dense(m, self.mem_units, use_bias=False, name="wah")))
        alphas = tf.squeeze(tf.exp(tf.keras.layers.dense(ha, units=1, use_bias=False, name='way')), axis=[-1])
        alphas = tf.math.divide(alphas, tf.math.reduce_sum(alphas, axis=0, keepdims=True))  # (max_time, batch_size)

        w_context = tf.math.reduce_sum(tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        # Late fusion
        lfc = tf.keras.layers.dense(w_context, self.num_units, use_bias=False, name='wfc')  # late fused context

        fw = tf.sigmoid(tf.keras.layers.dense(lfc, self.num_units, use_bias=False, name='wff') +
                        tf.keras.layers.dense(h, self.num_units, name='wfh')) # fusion weights
        
        hft = tf.math.multiply(lfc, fw) + h  # weighted fused context + hidden state
        
        return hft, new_state
