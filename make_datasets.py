import pandas as pd
import tensorflow as tf
import transformers as ppb
from sklearn.model_selection import train_test_split

df = pd.read_csv('books_v1_02_cleaned.tsv.gz', compression='gzip', sep='\t',nrows=400)
#df = pd.read_csv('out.csv')

df = df.astype({'star_rating': float})
df['star_rating'] = (df['star_rating'] - 1) / 4.

train_texts, val_texts, train_labels, val_labels = train_test_split(df.review_body.values, df.star_rating.values,
                                                                    test_size=.2)
test_texts, val_texts, test_labels, val_labels = train_test_split(val_texts, val_labels, test_size=.5)

"""Initialize pretrained DistilBertForSequenceClassification model and tokenizer. Change config for 5 output classifications. """

model_class, tokenizer_class, pretrained_weights = (
    ppb.TFBertForSequenceClassification, ppb.BertTokenizerFast,
    'bert-base-uncased')  # for trainer API

config = ppb.BertConfig.from_pretrained(pretrained_weights, num_labels=1, problem_type="regression")
tokenizer = tokenizer_class.from_pretrained(pretrained_weights, config=config)

train_encodings = tokenizer(train_texts.tolist(), padding=True, truncation=True)
val_encodings = tokenizer(val_texts.tolist(), padding=True, truncation=True)
test_encodings = tokenizer(test_texts.tolist(), padding=True, truncation=True)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))
for i in range(8):
    shard = train_dataset.shard(8, i)
    tf.data.experimental.save(shard, f"./dataset_shards/train_dataset_{i}", compression="GZIP")

    shard = val_dataset.shard(8, i)
    tf.data.experimental.save(shard, f"./dataset_shards/val_dataset_{i}", compression="GZIP")

    shard = test_dataset.shard(8, i)
    tf.data.experimental.save(shard, f"./dataset_shards/test_dataset_{i}", compression="GZIP")
