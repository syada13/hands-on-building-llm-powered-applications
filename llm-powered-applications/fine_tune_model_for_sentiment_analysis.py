
'''
Training Dataset - leverage the datasets library available in Hugging Face to load a binary classification dataset called IMDB.

The dataset contains movie reviews, which are classified as positive or negative. More specifically, the dataset contains two columns:

Text: The raw text movie review.
Label: The sentiment of that review. It is mapped as “0” for “Negative” and “1” for “Positive.”

Hugging Face datasets schema:

  DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})

'''

from datasets import load_dataset
dataset = load_dataset("imdb")
print(dataset["train"][100])

#Pre-process the dataset 
  #1. Tokenize the data. I am using HuggingFace utility 'AutoTokenizer' available in the Hugging Face Transformers library.

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-case")

# Below function format the dataset
def tokenize_function(examples):
  return tokenizer(examples["text"],padding = "max_length", truncation=True)

# Apply method to tokenize dataset
tokenized_datasets = dataset.map(tokenize_function,batched=True)
print(tokenized_datasets["train"][100]['input_ids'])

#Optionally, We can decide to reduce the size of your dataset if we want to make the training time shorter. In my case, I’ve shrunk the dataset into two sets – one for training, one for testing – of 500 observations each.

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

# Instatinate the BERT base model
import torch
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-based-case", num_labels=2)

#Evaluation - We will only use accuracy metric to evaluate our model.
import numpy as np
import evaluate
metric = evaluate.load("accuracy")

#function that computes the accuracy given the output of the training phase:
def compute_metrics(evaluate_prediction):
  logits,labels = evaluate_prediction
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# Set evaluation strategy, which means how often we want our model to be tested against the test set while training:
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs = 2,evaluation_strategy ="epoch")

#Trainer object is a class that provides an API for feature-complete training and evaluation of models in PyTorch, optimized for Hugging Face Transformers. 
trainer = Trainer(
  model = model,
  args= training_args,
  train_dataset = small_train_dataset,
  eval_dataset=small_eval_dataset,
  compute_metrics= compute_metrics
)

#Initiate the process of fine-tuning by calling the trainer
trainer.train()

#Once the model is trained, we can save it locally
trainer.save_model('model/sentiment-classifier')

#To consume and test the model, we can load it 
model = AutoModelForSequenceClassification.from_pretrained('models/sentiment-classifier')

# Model testing - Pass a sentence to the model (to be first tokenized) on which it can perform sentiment classification

inputs = tokenizer("I cannot stand it anymore!", return_tensors="pt")
outputs = model(**inputs)
print(outputs)

#Output - SequenceClassifierOutput(loss=None, logits=tensor([[ 0.6467, -0.0041]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)

#Since we are working with tensors, we will need to leverage the tensorflow library in Python. Plus, we will use the softmax function to obtain the probability vector associated with each label, so that we know that the final result corresponds to the label with the greatest probability:

import tensorflow as tf
predictions = tf.math.softmax(outputs.logits.detach(), axis=-1)
print(predictions)

#Output - tf.Tensor([[0.6571879  0.34281212]], shape=(1, 2), dtype=float32)
#Interpret output : Positive : If the sentiment score above 80 else Negative







