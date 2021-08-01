import tensorflow as tf
from tensorflow import TensorSpec
import logging
import os
import horovod.tensorflow.keras as hvd
import transformers as ppb
import numpy as np
from official.nlp import optimization

DATASET_TENSORSPEC = ({'input_ids': TensorSpec(shape=(512,), dtype=tf.int32, name=None),
                       'token_type_ids': TensorSpec(shape=(512,), dtype=tf.int32, name=None),
                       'attention_mask': TensorSpec(shape=(512,), dtype=tf.int32, name=None)},
                      TensorSpec(shape=(), dtype=tf.float64, name=None))

BATCH_SIZE = 16
N_EPOCHS = 3
OUTPUT_DIR = "./output"
IS_REGRESSION = True

logger = logging.getLogger(__name__)


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.output_dir.format(epoch=epoch))


def get_full_model(bert_model, regression=False):
    bert_config = bert_model.config

    weight_initializer = tf.keras.initializers.GlorotNormal()
    input_ids_layer = tf.keras.layers.Input(shape=(512,),
                                            name='input_ids',
                                            dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(512,),
                                                  name='attention_mask',
                                                  dtype='int32')
    last_hidden_state = bert_model([input_ids_layer, input_attention_layer])[0]

    cls_token = last_hidden_state[:, 0, :]

    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
    output = tf.keras.layers.Dense(bert_config.dim,
                                   kernel_initializer=weight_initializer,
                                   activation=leaky_relu)(cls_token)
    output = tf.keras.layers.Dropout(0.30)(output)
    output = tf.keras.layers.Dense(bert_config.num_labels,
                                   kernel_initializer=weight_initializer,
                                   kernel_constraint=None,
                                   bias_initializer='zeros',
                                   activation='linear' if regression else 'softmax')(output)
    full_model = tf.keras.Model([input_ids_layer, input_attention_layer], output)
    return full_model


def get_train_dataset(rank, lim=None):
    train_dataset = tf.data.experimental.load(f"./dataset_shards/train_dataset_{rank}",
                                              element_spec=DATASET_TENSORSPEC,
                                              compression="GZIP")
    if lim:
        train_dataset = train_dataset.take(lim)
    return train_dataset


# Function to map the labels from a regression suitable 0.0-1.0 to a classification suitable 0-4
def process_into_classification(dataset):
    def data_map(features, label):
        return features, int(label * 4)

    dataset = dataset.map(data_map, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def add_samples_weights(dataset, keys_tensor, values_tensor, precision=None):
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys_tensor,
            values_tensor,
            key_dtype=tf.string,
            value_dtype=tf.float32
        ),
        0
    )

    def data_map(features, label):
        weight = table.lookup(tf.strings.as_string(label, precision=precision))
        return features, label, weight

    dataset = dataset.map(data_map, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def get_val_dataset(rank, lim=None):
    val_dataset = tf.data.experimental.load(f"./dataset_shards/val_dataset_{rank}",
                                            element_spec=DATASET_TENSORSPEC,
                                            compression="GZIP")
    if lim:
        val_dataset = val_dataset.take(lim)
    return val_dataset


def get_test_dataset(num, lim=None):
    test_dataset = tf.data.experimental.load("./dataset_shards/test_dataset_0", element_spec=DATASET_TENSORSPEC,
                                             compression="GZIP")
    for i in range(1, num):
        in_data = tf.data.experimental.load(f"./dataset_shards/test_dataset_{i}", element_spec=DATASET_TENSORSPEC,
                                            compression="GZIP")
        test_dataset = test_dataset.concatenate(in_data)

    if lim:
        test_dataset = test_dataset.take(lim)

    return test_dataset


def main():
    # region Horovod initialization boilerplate
    hvd.init()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    # endregion

    # region Dataset loading and processing
    if IS_REGRESSION:
        keys_tensor = tf.constant(['0.00', '0.25', '0.50', '0.75', '1.00'])
    else:
        keys_tensor = tf.constant(['0', '1', '2', '3', '4'])
    values_tensor = tf.constant(
        [0.25516214033513884,
         0.3653295143642076,
         0.2432143530403531,
         0.10369791403102925,
         0.03259607822927125])
    train_dataset = get_train_dataset(hvd.rank())
    if not IS_REGRESSION:
        train_dataset = process_into_classification(train_dataset)
    train_dataset = add_samples_weights(train_dataset, keys_tensor, values_tensor, precision=2 if IS_REGRESSION else None)

    val_dataset = get_val_dataset(hvd.rank())
    if not IS_REGRESSION:
        val_dataset = process_into_classification(val_dataset)

    if hvd.rank() == 0:
        test_dataset = get_test_dataset(hvd.size())
        if not IS_REGRESSION:
            test_dataset = process_into_classification(test_dataset)
    # endregion

    # region Model compilation with optimizer and loss
    num_train_samples = tf.data.experimental.cardinality(train_dataset).numpy()
    num_train_steps = N_EPOCHS * (num_train_samples // BATCH_SIZE)
    num_warmup_steps = int(0.1 * num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=2e-5 * hvd.size(),
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    optimizer = hvd.DistributedOptimizer(optimizer)

    if IS_REGRESSION:
        config = ppb.DistilBertConfig(output_hidden_states=True, num_labels=1)
        loss_fn = tf.keras.losses.MeanSquaredError()
        metrics = []
    else:
        config = ppb.DistilBertConfig(output_hidden_states=True, num_labels=5)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    bert_model = ppb.TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
    full_model = get_full_model(bert_model, regression=IS_REGRESSION)

    full_model.compile(optimizer=optimizer,
                       loss=loss_fn,
                       metrics=metrics,
                       experimental_run_tf_function=False)
    # endregion

    # region Model fitting
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

    if hvd.rank() == 0:
        callbacks.append(SavePretrainedCallback(output_dir="./checkpoints/checkpoint-test-regres-{epoch}"))

    num_val_samples = tf.data.experimental.cardinality(val_dataset).numpy()

    full_model.fit(train_dataset.shuffle(10000).repeat().batch(BATCH_SIZE),
                   validation_data=val_dataset.shuffle(10000).repeat().batch(BATCH_SIZE),
                   epochs=N_EPOCHS,
                   callbacks=callbacks,
                   verbose=1 if hvd.rank() == 0 else 0,
                   steps_per_epoch=num_train_samples // BATCH_SIZE,
                   validation_steps=num_val_samples // BATCH_SIZE)
    # endregion

    # region Model inference
    if hvd.rank() == 0:
        logger.info("Predictions on test dataset...")
        num_test_steps = tf.data.experimental.cardinality(test_dataset).numpy() // BATCH_SIZE
        predictions = full_model.predict(test_dataset.batch(BATCH_SIZE), steps=num_test_steps, verbose=1)
        predicted_class = np.squeeze(predictions) if IS_REGRESSION else np.argmax(predictions, axis=1)
        out_test_file = os.path.join(OUTPUT_DIR, "test_results_regres.txt")
        with open(out_test_file, "w") as writer:
            writer.write(str(predicted_class.tolist()))
            for ele in test_dataset.enumerate().as_numpy_iterator():
                writer.write(str(ele))
            logger.info(f"Wrote predictions to {out_test_file}")
    # endregion


if __name__ == "__main__":
    main()
