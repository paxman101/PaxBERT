import tensorflow as tf
from tensorflow import TensorSpec
import transformers as ppb
import argparse
import os
import numpy as np

DATASET_TENSORSPEC = ({'input_ids': TensorSpec(shape=(512,), dtype=tf.int32, name=None),
                       'token_type_ids': TensorSpec(shape=(512,), dtype=tf.int32, name=None),
                       'attention_mask': TensorSpec(shape=(512,), dtype=tf.int32, name=None)},
                      TensorSpec(shape=(), dtype=tf.float64, name=None))

OUTPUT_DIR = "./output"

parser = argparse.ArgumentParser(description="Run predictions with dataset with given saved model.")
parser.add_argument("model_dir", type=str)
parser.add_argument("-b", "--batch-size", dest="BATCH_SIZE", default=16)

model_class, tokenizer_class, pretrained_weights = (
    ppb.TFDistilBertModel, ppb.BertTokenizerFast, 'distilbert-base-uncased')  # for trainer API

config = ppb.DistilBertConfig(output_hidden_states=True, num_labels=1)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights, config=config)
model = model_class.from_pretrained(pretrained_weights, config=config)

weight_initializer = tf.keras.initializers.GlorotNormal()
input_ids_layer = tf.keras.layers.Input(shape=(512,),
                                        name='input_ids',
                                        dtype='int32')
input_attention_layer = tf.keras.layers.Input(shape=(512,),
                                              name='attention_mask',
                                              dtype='int32')
last_hidden_state = model([input_ids_layer, input_attention_layer])[0]

cls_token = last_hidden_state[:, 0, :]

leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
output = tf.keras.layers.Dense(config.dim,
                               kernel_initializer=weight_initializer,
                               activation=leaky_relu)(cls_token)
output = tf.keras.layers.Dropout(0.30)(output)
output = tf.keras.layers.Dense(config.num_labels,
                               kernel_initializer=weight_initializer,
                               kernel_constraint=None,
                               bias_initializer='zeros')(output)
full_model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

args = parser.parse_args()

model_dir = args.model_dir
BATCH_SIZE = args.BATCH_SIZE

full_model.load_weights(model_dir)

test_dataset = tf.data.experimental.load("./dataset_shards/test_dataset_0", element_spec=DATASET_TENSORSPEC, compression="GZIP")
for i in range(1, 8):
    in_data = tf.data.experimental.load(f"./dataset_shards/test_dataset_{i}", element_spec=DATASET_TENSORSPEC, compression="GZIP")
    test_dataset = test_dataset.concatenate(in_data)

predictions = full_model.predict(test_dataset.batch(batch_size=BATCH_SIZE), verbose=1)
predicted_class = np.squeeze(predictions)
out_test_file = os.path.join(OUTPUT_DIR, "test_results_train.txt")
with open(out_test_file, "w") as writer:
    writer.write(str(predicted_class.tolist()))
    for ele in test_dataset.enumerate().as_numpy_iterator():
        writer.write(str(ele))
