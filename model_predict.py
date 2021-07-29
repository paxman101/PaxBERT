import tensorflow as tf
from tensorflow import TensorSpec
import transformers as ppb
import argparse
import os

DATASET_TENSORSPEC = ({'input_ids': TensorSpec(shape=(512,), dtype=tf.int32, name=None),
                       'token_type_ids': TensorSpec(shape=(512,), dtype=tf.int32, name=None),
                       'attention_mask': TensorSpec(shape=(512,), dtype=tf.int32, name=None)},
                      TensorSpec(shape=(), dtype=tf.float64, name=None))

OUTPUT_DIR = "./output"

parser = argparse.ArgumentParser(description="Run predictions with dataset with given saved model.")
parser.add_argument("model_dir", type=str)
parser.add_argument("-b", "--batch-size", dest="BATCH_SIZE", default=16)

args = parser.parse_args()

model_dir = args.model_dir
BATCH_SIZE = args.BATCH_SIZE
print(model_dir)

config = ppb.DistilBertConfig.from_pretrained('./checkpoint-1.h5', num_labels=1, problem_type="regression")

test_dataset = tf.data.experimental.load("./dataset_shards/test_dataset_0", element_spec=DATASET_TENSORSPEC, compression="GZIP")
#for i in range(1, 8):
#    in_data = tf.data.experimental.load(f"./dataset_shards/test_dataset_{i}", element_spec=DATASET_TENSORSPEC, compression="GZIP")
#    test_dataset = test_dataset.concatenate(in_data)

model = ppb.TFDistilBertForSequenceClassification.from_pretrained(model_dir, config=config)

predictions = model.predict(test_dataset.batch(batch_size=BATCH_SIZE))
out_test_file = os.path.join(OUTPUT_DIR, "test_results.txt")
with open(out_test_file, "w") as writer:
    writer.write(str(predictions.to_tuple()))
    for ele in test_dataset.enumerate().as_numpy_iterator():
        writer.write(str(ele))
