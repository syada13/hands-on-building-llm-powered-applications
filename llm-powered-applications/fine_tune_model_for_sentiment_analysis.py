
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









