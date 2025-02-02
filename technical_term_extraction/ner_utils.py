"""
This file contains the utility functions for the named entity recognition fine-tuning and data handling.
"""

import pandas as pd
import evaluate
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset, ClassLabel
from copy import deepcopy
import numpy as np
import os
import json

def conll_to_dataframe(annotaions_dir: str) -> tuple:
    """
    The function iterates over all the annotation csv files, reads each file line by line and creates a 
    big dataset dictionary. It is then further processed into a train dataset and eval dataset. Both in
    pandas DataFrame-format.

    Args:
        - annotations_dir (str): The path to the annotations directory
    
    Output:
        - tuple that contains the train-split and eval-split of the datasets
    """
    dataset = []
    for filename in os.listdir(annotaions_dir):
        file_path = os.path.join(annotaions_dir, filename)

        if filename == ".ipynb_checkpoints":
            continue
        
        with open(file_path, "r", encoding="utf-8") as file:
            text = []
            labels = []
            for line in file:

                if "DOCSTART" in line:
                    continue

                if len(line.replace("\n", "")) == 0:

                    dataset.append(deepcopy({"tokens": text, "ner_tags": labels}))
                    text = []
                    labels = []
                    continue

                split_line = line.split(" -X- _ ")
                token = split_line[0]
                label = split_line[-1]

                if "O" in label:
                    label = "O"
                else:
                    label = label[2:].replace("\n", "")

                text.append(token)
                labels.append(label)

    dataset = pd.DataFrame(dataset).sample(frac=1, random_state=1502).reset_index(drop=True)
    len_test = int(len(dataset) * 0.2)
    len_train = int(len(dataset) - len_test)

    eval_dataset = dataset[:len_test]
    train_dataset = dataset[len_test:]

    return train_dataset, eval_dataset

def load_dataset(annotations_dir: str, tokenizer: 'tokenizer', classmap, overfit) -> tuple:
    """
    The function takes the path to the csv annotations and creates the train and eval datasets
    in the format huggingface transformers expect

    Args:
        - annotations_dir (str): The path to the annotations directory
    
    Output:
        - train_dataset (arrow_dataset.Dataset): The train-split of the dataset in a arrow_dataset.Dataset
        - eval_dataset (arrow_dataset.Dataset): The eval-split of the dataset in a arrow_dataset.Dataset
    """

    train_dataset, eval_dataset = conll_to_dataframe(annotations_dir)

    if overfit:
        train_dataset, eval_dataset = train_dataset[0:3], eval_dataset[:1]

    train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_dataset))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_dataset))

    train_dataset = train_dataset.map(lambda x: tokenizer(x["tokens"], truncation=True, is_split_into_words=True))
    eval_dataset = eval_dataset.map(lambda x: tokenizer(x["tokens"], truncation=True, is_split_into_words=True))

    train_dataset = train_dataset.map(lambda y: {"ner_tags": classmap.str2int(y["ner_tags"])})
    eval_dataset = eval_dataset.map(lambda y: {"ner_tags": classmap.str2int(y["ner_tags"])})

    return train_dataset, eval_dataset

def compute_metrics(p: tuple) -> dict:
    """
    This function is needed for the huggingface trainer object. 

    Args
        - p (tuple): A tuple that contains the predictions, and the labels

    Output
        - Dictionary that contains the evaluation metrics
    """
    label_list = ["O", "I-TECT", "B-TECT"]
    seqeval = evaluate.load("seqeval")

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
    """
    This function prepares tokenized inputs and aligns labels for Named Entity Recognition.
    It ensures that labels are aligned with tokenized words by setting special tokens and non-initial tokens 
    of each word to -100, which will be ignored during model training.

    Args:
        - examples (dict): A dictionary containing the input data. The key "tokens" refers to the list of word tokens, 
                           and the key "ner_tags" refers to the corresponding labels.
        - tokenizer: A tokenizer object used to tokenize the input examples.

    Output:
        - Dictionary containing tokenized inputs with an additional "labels" field, which has aligned labels.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def save_inference(sentence: str, results: list, gt_labels: list, out_path: str, lstm=False) -> None:
    """
    This function saves the sentence, the predicted labels and the ground truth labels in a json file.

    Args:
        - sentence (str): The sentence that might contain technical terms
        - results (list): The predictions of the named entity recognition model
        - gt_labels (list): The ground truth labels to the sentence
        - out_path (str): The path, where the predictions will be saved
        - lstm (bool): Whether the predictions are from an LSTM or from the BERT models

    Output:
        - None
    """
    if lstm:
        complete_inference = {
            "sentence": sentence,
            "results": results,
            "gt_labels": gt_labels
        }

    else:
        formatted_results = []
        for result in results:
          end = result["start"]+len(result["word"].replace("##", ""))  
          
          if result["word"].startswith("##"):
            formatted_results[-1]["end"] = end
            formatted_results[-1]["word"]+= result["word"].replace("##", "")
          else:
            formatted_results.append({
                'start': result["start"], 
                'end': end,
                'entity': result["entity"],
                'index': result["index"],
                'score': result["score"],
                'word': result["word"]})
    
        complete_inference = {
            "sentence": sentence,
            "results": formatted_results,
            "gt_labels": gt_labels
        }

    with open(out_path, 'w') as outfile:
        json.dump(str(complete_inference), outfile)
    
    return None