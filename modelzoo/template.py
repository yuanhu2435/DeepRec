import argparse
import tensorflow as tf
import os

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']


class WDL():
    def __init__(self):
        self._label = None
        self._feature = None
        self.bf16 = False
        self._is_training = True

        self._create_model()
        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    # create model
    def _create_model(self):
        self.logits = None
        self.probability = tf.math.sigmod(self.logits)
        self.output = tf.round(self.probability)

    # compute loss
    def _create_loss(self):
        self.loss = None

    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.train_op = None

    # compute acc & auc
    def _create_metrics(self):
        self.acc = None
        self.auc = None
        self._add_layer_summary(self.acc)


# generate dataset pipline
def build_model_input():
    pass


# generate feature columns
def build_feature_columns():
    pass


def train():
    while not sess.should_stop():
        sess.run(
            [model.loss, model.train_op]

    print('gsteps,loss,steps')


def eval():
    for _in in 'test steps':
        session.run([model.acc_op,model.auc_op])

    print('AUC,ACC')


def main(tf_config=None, server=None):
    # check dataset and count data set size
    print('Dataset')

    # set batch size, eporch & steps
    print('Batch size, epoch, steps')

    # set fixed random seed
    print('Random seed')

    # set directory path for checkpoint_dir
    print('Checkpoint directory path')

    # create data pipline of train & test dataset
    train_dataset = build_model_input()
    test_dataset = build_model_input()

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create feature column
    feature_column = build_feature_columns()

    # create variable partitioner for distributed training
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    input_layer_partitioner = None
    dense_layer_partitioner = None

    # create model
    model = WDL()

    # Session config
    sess_config = tf.ConfigProto()

    # Session hook
    hooks = []

    # Run model training and evaluation
    train()
    eval()


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    return parser


# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    pass


# Some DeepRec's features are enabled by ENV.
# This func is used to set ENV and enable these features.
# A triple quotes comment is used to introduce these features and play an emphasizing role.
def set_env_for_DeepRec():
    '''
    DNNL_MAX_CPU_ISA: Specify the highest instruction set used by oneDNN (when the version is less than 2.5.0), 
        it will be set to AVX512_CORE_AMX to enable Intel CPU's feature.
    '''
    pass


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    if 'DeepRec':
        set_env_for_DeepRec()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        tf_config, server = generate_cluster_info(TF_CONFIG)
        main(tf_config, server)
