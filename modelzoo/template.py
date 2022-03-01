import argparse
import tensorflow as tf
import os

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["clicked"]


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


# generate dataset pipline
def generate_input_data():
    pass


# generate feature columns
def build_feature_cols():
    pass


class WDL():
    def __init__(self):
        self.label = None
        self.feature = None
        self.bf16 = False
        self._is_training = True

        self.input_fn()
        self.model_fn()
        self.train_fn()
        self.metrics_fn()

    # process input data
    def input_fn(self):
        pass

    # model structure
    # finally get a probability of predict
    def model_fn(self):
        self.probability = None
        pass

    # compute loss and define optimizer
    def train_op(self):
        self.train_op = None
        pass

    # compute acc & auc
    def metrics_fn(self):
        self.acc = None
        self.auc = None
        pass

    def train(self, session, timeline=False, tensorboard=False):
        self._is_training = True
        if tensorboard:
            pass
        if timeline:
            pass

        session.run(self.probability, self.train_op)

    def eval(self, session):
        self._is_training = False
        acc, auc = session.run(self.acc, self.auc)
        return acc, auc

    # save checkpoint
    def save(self, session):
        pass


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    return parser


def train(session, model, dataset, steps):
    for _in in 'train steps':
        if 'tensorborad events':
            model.train(session, tensorborad=True)
        elif 'timeline':
            model.train(session, timeline=True)
        else:
            model.train(session)

        if 'save checkpoint':
            model.save(session)

        print('gsteps,loss,steps')
    pass


def eval(session, model, dataset, steps):
    model.eval()
    print('AUC,ACC')
    pass


def main(tf_config=None, server=None):
    # check dataset and count data set size
    print('Dataset')

    # set batch size, eporch & steps
    print('Batch size, epoch, steps')

    # set fixed random seed
    print('Random seed')

    # set directory path for checkpoint_dir
    print('Checkpoint directory path')

    # create feature column
    feature_column = build_feature_cols()

    # create data pipline of train & test dataset
    train_dataset = generate_input_data()
    test_dataset = generate_input_data()

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

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

    # Session run()
    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           config=sess_config) as sess:
        train(sess, model, train_init_op, 'train_steps')
        eval(sess, model, test_init_op, 'test_steps')


# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    pass


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        tf_config, server = generate_cluster_info(TF_CONFIG)
        main(tf_config, server)
