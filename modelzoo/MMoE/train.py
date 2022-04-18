import time
import argparse
import numbers
import tensorflow as tf
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json

from tensorflow.python.ops import partitioned_variables

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))
SEED=2021
'''
INPUT CONFIG SPECIFICS
'''
TRAIN_DATA_NAME="taobao_train_data"
TEST_DATA_NAME="taobao_test_data"

LABEL_COLUMNS = ["clk", "buy"]
HASH_INPUTS = [
    "pid",
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
    "user_id",
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
    "tag_category_list",
    "tag_brand_list"
    ]
IDENTITY_INPUTS = ["price"]
ALL_FEATURE_COLUMNS = HASH_INPUTS + IDENTITY_INPUTS
ALL_INPUT = LABEL_COLUMNS + HASH_INPUTS + IDENTITY_INPUTS
NOT_USED_CATEGORY = ["final_gender_code"]

HASH_BUCKET_SIZES = {
    'pid': 10,
    'adgroup_id': 100000,
    'cate_id': 10000,
    'campaign_id': 100000,
    'customer': 100000,
    'brand': 100000,
    'user_id': 100000,
    'cms_segid': 100,
    'cms_group_id': 100,
    'final_gender_code': 10,
    'age_level': 10,
    'pvalue_level': 10,
    'shopping_level': 10,
    'occupation': 10,
    'new_user_class_level': 10,
    'tag_category_list': 100000,
    'tag_brand_list': 100000
}

NUM_BUCKETS = {
    'price': 50
}

def generate_taobao_input_data(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        HASH_defaults = [[" "] for i in range(0, len(HASH_INPUTS))]
        label_defaults = [[0] for i in range (0, len(LABEL_COLUMNS))]
        IDENTITY_defaults = [[0] for i in range(0, len(IDENTITY_INPUTS))]
        column_headers = LABEL_COLUMNS + HASH_INPUTS + IDENTITY_INPUTS
        record_defaults = label_defaults + HASH_defaults + IDENTITY_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = []
        for i in range(0, len(LABEL_COLUMNS)):
            labels.append(all_columns.pop(LABEL_COLUMNS[i]))
        label = tf.stack(labels, axis=1)
        features = all_columns
        return features, label

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.shuffle(buffer_size=400000,
                              seed=SEED)  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(32)
    return dataset

def build_feature_cols():
    feature_cols = []
    for column_name in ALL_FEATURE_COLUMNS:
        if column_name in NOT_USED_CATEGORY:
            continue
        if column_name in HASH_INPUTS:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                dtype=tf.string)

            feature_cols.append(
                tf.feature_column.embedding_column(categorical_column,
                                                   dimension=16,
                                                   combiner='mean'))
        elif column_name in IDENTITY_INPUTS:
            column = tf.feature_column.categorical_column_with_identity(column_name, 50)
            feature_cols.append(
                tf.feature_column.embedding_column(column,
                                                   dimension=16,
                                                   combiner='mean'))
        else:
            raise ValueError('Unexpected column name occured')

    return feature_cols

'''
END OF INPUT CONFIG SPECIFICS
'''
'''
MODEL CONFIG SPECIFICS
'''
DNN_ACTIVATION = tf.nn.relu

L2_REGULARIZATION = 1e-06
EMBEDDING_REGULARIZATION = 5e-05

EXPERTS_COUNT = 4
EXPERT_HIDDEN_UNITS = [256, 192, 128, 64]

#Tower tuple structure (tower name, label name, hidden units)
TOWERS = [
    ("ctr", "clk", [256, 192, 128, 64]),
    ("cvr", "buy", [256, 192, 128, 64])
]
'''
MODEL CONFIG SPECIFICS
'''
def make_scope(name, bf16):
    if(bf16):
        return tf.variable_scope(name, reuse=tf.AUTO_REUSE).keep_weights()
    else:
        return tf.variable_scope(name, reuse=tf.AUTO_REUSE)

def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)

def l2_regularizer(scale, scope=None):
  if isinstance(scale, numbers.Integral):
    raise ValueError(f'Scale cannot be an integer: {scale}')
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError(f'Setting a scale less than 0 on a regularizer: {scale}.')
    if scale == 0.:
      return lambda _: None

  def l2(weights):
    with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
      my_scale = tf.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
      return tf.math.multiply(my_scale, tf.nn.l2_loss(weights), name=name)

  return l2

class MMOE():
    def __init__(self,
                 input,
                 feature_column,
                 num_experts,
                 expert_hidden_units,
                 towers,
                 l2_scale,
                 learning_rate=0.1,
                 bf16=False,
                 is_training=True):
        self.__feature_column = feature_column
        self.__num_experts = num_experts
        self.__expert_hidden_units = expert_hidden_units
        self.__towers = towers
        self.__l2_regularization = l2_regularizer(l2_scale) if l2_scale else None
        self.__learning_rate = learning_rate
        self.__bf16 = bf16
        self.__is_training = is_training

        self.feature = input[0]
        self.label = input[1]

        self.model = self.__build_model()

        with tf.name_scope('head'):
            self.train_op, self.loss = self.compute_loss()
            self.acc, self.acc_op = tf.metrics.accuracy(labels=self.label, predictions=tf.round(self.model))
            self.auc, self.auc_op = tf.metrics.auc(labels=self.label, predictions=self.model, num_thresholds=1000)

            tf.summary.scalar('eval_acc', self.acc)
            tf.summary.scalar('eval_auc', self.auc)

    def compute_loss(self):
        bce_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.model = tf.squeeze(self.model)
        loss = tf.math.reduce_mean(bce_loss_func(self.label, self.model))
        tf.summary.scalar('loss', loss)

        # TODO: add decay
        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.__learning_rate)

        train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op, loss

    def __build_model(self):
        TAG_COLUMN = ['tag_category_list', 'tag_brand_list']
        for key in TAG_COLUMN:
            self.feature[key] = tf.strings.split(self.feature[key], '|')

        with make_scope('input_layer', self.__bf16):
            input_emb = tf.feature_column.input_layer(self.feature,
                                                      self.__feature_column)
        
        with make_scope('MMOE', self.__bf16):
            if self.__bf16:
                input_emb = tf.cast(input_emb, dtype=tf.bfloat16)
            experts = []
            for i in range(1, self.__num_experts + 1):
                with tf.variable_scope(f'expert_{i}'):
                    expert_features = input_emb

                    for layer_id, num_hidden_units in enumerate(self.__expert_hidden_units):
                        with tf.variable_scope(f'expert_{i}_layer_{layer_id}', reuse=tf.AUTO_REUSE) as expert_layer_scope:
                            expert_features = tf.layers.dense(expert_features,
                                                            units=num_hidden_units,
                                                            activation=None,
                                                            kernel_regularizer=self.__l2_regularization,
                                                            name=f'{expert_layer_scope.name}/dense')
                            expert_features = DNN_ACTIVATION(expert_features, 
                                                            name=f'{expert_layer_scope.name}/act')               
                            add_layer_summary(expert_features, expert_layer_scope.name)
                    experts.append(expert_features)
            experts_features = tf.stack(experts, axis=1)

            towers=[]
            for [tower_name, label_name, hidden_units] in self.__towers:
                gate_input = input_emb
                with tf.variable_scope(f'{tower_name}_gate', reuse=tf.AUTO_REUSE) as gate_scope:

                    gate = tf.layers.dense(input_emb,
                                        units=self.__num_experts,
                                        kernel_regularizer=self.__l2_regularization,
                                        name=f'{tower_name}_gate')
                    gate = tf.nn.softmax(gate, axis=1)
                    gate = tf.expand_dims(gate, -1)

                with tf.variable_scope(tower_name):
                    tower_input = expert_features

                    tower_input = tf.multiply(experts_features, gate)
                    tower_input = tf.reduce_sum(tower_input, axis=1)

                    tower_features = tower_input
                    for layer_id, num_hidden_units in enumerate(hidden_units):
                        with tf.variable_scope(f'{tower_name}_layer_{layer_id}', reuse=tf.AUTO_REUSE) as tower_layer_scope:
                            tower_features = tf.layers.dense(tower_features,
                                                            units=num_hidden_units,
                                                            kernel_regularizer=self.__l2_regularization,
                                                            name=f'{tower_layer_scope.name}/dense')
                            tower_features = DNN_ACTIVATION(tower_features,
                                                            name=f'{tower_layer_scope.name}/act')
                            add_layer_summary(tower_features, tower_layer_scope.name)
                    final_tower_predict = tf.layers.dense(inputs=tower_features,
                                                        units=1,
                                                        activation=None,
                                                        kernel_regularizer=self.__l2_regularization,
                                                        name=f'{tower_name}_output')
                    final_tower_predict = tf.math.sigmoid(final_tower_predict, f'{tower_name}_output')
                    towers.append(final_tower_predict)
            tower_stack = tf.stack(towers,axis=1)
            if self.__bf16:
                tower_stack = tf.cast(tower_stack, dtype=tf.float32)

        return tower_stack

def train(model,
          checkpoint_dir,
          train_dataset,
          test_dataset,
          keep_checkpoint_max,
          train_steps,
          test_steps,
          no_eval=False):
    model_dir = os.path.join(args.output_dir, 'model_MMOE_' + str(int(time.time())))
    checkpoint_dir = checkpoint_dir if checkpoint_dir else model_dir
    print(f'Saving model checkpoints to {checkpoint_dir}')

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    sess_config = tf.ConfigProto()
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=keep_checkpoint_max)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # train model
        sess.run(train_init_op)

        start = time.perf_counter()
        for _in in range(0, train_steps):
            if args.save_steps > 0 and (_in % args.save_steps == 0
                                        or _in == train_steps - 1):
                _, train_loss, events = sess.run([model.train_op, model.loss, merged])
                writer.add_summary(events, _in)
                checkpoint_path = saver.save(sess,
                                             save_path=os.path.join(
                                             checkpoint_dir,
                                             'MMOE_checkpoint'),
                                             global_step=_in)
                print(f'Save checkpoint to {checkpoint_path}')
            elif (args.timeline > 0 and _in % args.timeline == 0):
                _, train_loss = sess.run([model.train_op, model.loss],
                                         options=options,
                                         run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                print(f'Save timeline to {checkpoint_path}')
                with open(os.path.join(checkpoint_dir,f'timeline-{_in}.json'), 'w') as f:
                    f.write(chrome_trace)
            else:
                _, train_loss = sess.run([model.train_op, model.loss])

            # print training loss and time cost
            if (_in % 100 == 0 or _in == train_steps - 1):
                end = time.perf_counter()
                cost_time = end - start
                global_step_sec = (100 if _in % 100 == 0 else train_steps -
                                   1 % 100) / cost_time
                print(f'global_step/sec: {global_step_sec:.4f}')
                print(f'loss = {train_loss}, steps = {_in}, cost time = {cost_time:0.2f}s')
                start = time.perf_counter()

        # eval model
        if not no_eval:
            writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'))

            sess.run(test_init_op)
            for _in in range(1, test_steps + 1):
                if (_in != test_steps):
                    sess.run([model.acc, model.acc_op, model.auc, model.auc_op])
                    if (_in % 1000 == 0):
                        print(f'Evaluation complete:[{_in}/{test_steps}]')
                else:
                    eval_acc, _, eval_auc, _, events = sess.run([model.acc,
                                                                 model.acc_op,
                                                                 model.auc,
                                                                 model.auc_op,
                                                                 merged])
                    writer.add_summary(events, _in)
                    print(f'Evaluation complete:[{_in}/{test_steps}]')
                    print(f'ACC = {eval_acc}\nAUC = {eval_auc}')

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--job',
                        help='train or test model',
                        type=str,
                        choices=["train", "test"],
                        default='train')
    parser.add_argument('--seed', help='set random seed', type=int, default=SEED)
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output directory',
                        required=False)
    parser.add_argument('--model_dir',
                        help='Full path to test model directory',
                        required=False)
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.1)
    parser.add_argument('--l2_regularization',
                        help='L2 regularization for the model',
                        type=float,
                        default=L2_REGULARIZATION)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline',
                        type=int,
                        default=0)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    # TODO: validate arguments: data location existence, checkpoint path existence

    SEED = args.seed
    tf.set_random_seed(SEED)

    print('Checking dataset')
    train_file = os.path.join(args.data_location, 'taobao_train_data')
    test_file = os.path.join(args.data_location, 'taobao_test_data')

    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        raise ValueError('taobao_train_data or taobao_test_data does not exist '
                         'in the given data_location. Please provide valid path')

    no_of_training_examples = sum(1 for _ in open(train_file))
    no_of_test_examples = sum(1 for _ in open(test_file))

    # set params
    # set batch size & steps
    batch_size = args.batch_size
    if args.steps == 0:
        no_epochs = 10
        train_steps = math.ceil(
            (float(no_epochs) * no_of_training_examples) / batch_size)
    else:
        no_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)

    feature_cols = build_feature_cols()
    train_dataset = generate_taobao_input_data(train_file, batch_size, no_epochs)
    test_dataset = generate_taobao_input_data(test_file, batch_size, 1)

    print(f'Numbers of training dataset: {no_of_training_examples}')
    print(f'Number of epochs: {no_epochs}')
    print(f'Number of train steps: {train_steps}')

    print(f'Numbers of test dataset: {no_of_test_examples}')
    print(f'Numbers of test steps: {test_steps}')

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    next_element = iterator.get_next()

    model = MMOE(next_element, feature_cols, EXPERTS_COUNT, EXPERT_HIDDEN_UNITS, TOWERS, L2_REGULARIZATION, bf16=args.bf16)
    # TODO: add TF_CONFIG info
    # TODO: add test feature
    # TODO: add paralelism
    if args.job == 'train':
        train(model,
              args.checkpoint,
              train_dataset,
              test_dataset,
              args.keep_checkpoint_max,
              train_steps,
              test_steps,
              args.no_eval)
    elif args.job == 'test':
        pass
    else:
        print('[ERROR] Incorrect job provided. Available jobs: train, test')
        sys.exit(1)
