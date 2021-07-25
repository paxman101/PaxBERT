"""NLP Project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1avJo4M_jIaZ0jw27aOVDSIRcDK385jyC

NLP with review classification
"""

from collections import Counter

import pandas as pd
import torch
import transformers as ppb
from sklearn.model_selection import train_test_split


df = pd.read_csv('books_v1_02_cleaned.tsv.gz', compression='gzip', sep='\t')
# df = pd.read_csv('out.csv')

df = df.astype({'star_rating': float})
df['star_rating'] = (df['star_rating'] - 1) / 4.

"""Split our dataset into a training set and an evaluation set"""

train_texts, val_texts, train_labels, val_labels = train_test_split(df.review_body.values, df.star_rating.values,
                                                                    test_size=.2)
test_texts, val_texts, test_labels, val_labels = train_test_split(val_texts, val_labels, test_size=.5)

model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertForSequenceClassification, ppb.DistilBertTokenizerFast, 'distilbert-base-uncased')  # for trainer API

config = ppb.DistilBertConfig.from_pretrained(pretrained_weights, num_labels=1, problem_type="regression")
tokenizer = tokenizer_class.from_pretrained(pretrained_weights, config=config)
base_model = model_class.from_pretrained(pretrained_weights, config=config)
base_model.train()

train_encodings = tokenizer(train_texts.tolist(), padding=True, truncation=True)
val_encodings = tokenizer(val_texts.tolist(), padding=True, truncation=True)
test_encodings = tokenizer(test_texts.tolist(), padding=True, truncation=True)

"""#Fine-tune BERT with dataset trainer API

Turn our encodings and labels into PyTorch Dataset objects.
"""


# https://huggingface.co/transformers/custom_datasets.htm
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)
test_dataset = ReviewDataset(test_encodings, test_labels)
# print(train_dataset.encodings.items)

'Use the Trainer API from HuggingFace to fine-tune the model. '

"""Define a subclass of the Huggingface Trainer which overwrites the get_train_dataloader and _get_train_sampler functions to replace their sampler with a weighted one to address class imbalance."""

# https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a


class ImbalancedTrainer(ppb.Trainer):
    def _get_train_sampler(self, train_dataset):
        class_count = [j for i, j in sorted(dict(Counter(train_dataset.labels)).items())]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        full_class_weights = class_weights[train_dataset.labels]
        #
        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            weights=full_class_weights,
            num_samples=len(full_class_weights),
            replacement=True,
        )
        #
        return weighted_sampler

    #
    def get_train_dataloader(self):
        train_dataset = self.train_dataset

        weighted_sampler = self._get_train_sampler(train_dataset)

        return torch.utils.data.DataLoader(
            train_dataset,
            sampler=weighted_sampler,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


training_args = ppb.TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=4,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
)
trainer = ImbalancedTrainer(
    model=base_model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
)
trainer.train()
trainer.save('./distilBertModel')

data = enumerate(trainer.get_train_dataloader())
labels = []
for batch, l in data:
    labels.extend(l['labels'].tolist())

trainer = ImbalancedTrainer(
    model=base_model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
)

data = enumerate(trainer.get_train_dataloader())
texts = []
labels = []
for batch, l in data:
    texts.extend(l['input_ids'].tolist())
    labels.extend(l['labels'].tolist())

predict = trainer.predict(test_dataset=test_dataset)
with open('predictions.txt', 'a') as file:
    file.write(predict)
