"""
This script contains the Hyperparameter optimization script for the training of the named entity recognition models.
"""

import torch
import gc
import optuna
from datasets import ClassLabel, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from visualization_utils import output_results
from optuna.samplers import RandomSampler, TPESampler

def run_hpo(checkpoint_tuple: tuple, lstm=False) -> None:
    """
    This is the main Hyperparameter optimization script. It does the HPO for both the bert models and the LSTM+CRF model

    Args:
        - checkpoint_tuple (tuple): A tuple that contains the model name and the model checkpoint
        - lstm (bool): A boolean, whether the LSTM model should be tuned or not
    
    Output:
        - None
    """
    checkpoint, model_name = checkpoint_tuple

    #Load tokenizer
    if model_name == "roberta_base" or model_name == "roberta_large":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    classmap = ClassLabel(num_classes=3, names=['O', 'I-TECT', 'B-TECT'])

    #Prepare dataset
    train_dataset, eval_dataset = load_dataset("annotations", tokenizer, classmap, overfit=False)

    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})

    #HPO for the LSTM+CRF
    if lstm:
        model_name = "lstm"

        #Prepare dataset
        columns = {0: 'text', 2: 'ner'}
        
        corpus: Corpus = ColumnCorpus("lstm_data", columns,
                                      train_file='train.txt',
                                      test_file='test.txt',
                                      dev_file='val.txt')
        
        label_type = 'ner'
    
        label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)

        #Objective function that does the HPO
        def objective(trial):
            #Define the parameters that need to be optimized
            learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.15])
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            hidden_size= trial.suggest_categorical('hidden_size',[128, 256, 512])
            rnn_layers= trial.suggest_categorical('rnn_layers',[1, 2, 4])
            dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
        
            #Choose embeddings for the LSTM
            embedding_types = [
                WordEmbeddings('glove'),
                FlairEmbeddings('news-forward'),
                FlairEmbeddings('news-backward'),
            ]
        
            embeddings = StackedEmbeddings(embeddings=embedding_types)
                                          
            tagger = SequenceTagger(hidden_size=hidden_size,
                                    rnn_layers=rnn_layers,
                                    embeddings=embeddings,
                                    tag_dictionary=label_dict,
                                    tag_type=label_type)
        
            trainer = ModelTrainer(tagger, corpus)
            
            result = trainer.train(f"lstm_hpo/trial_{trial.number}",
                      learning_rate=learning_rate,
                      mini_batch_size=batch_size,
                      max_epochs=100)
        
            return result['test_score']
        
        #Start HPO
        study = optuna.create_study(sampler=RandomSampler(), direction='maximize')
        study.optimize(objective, n_trials=200)
        
        print(f"Best trial for {model_name}:")
        best_trial = study.best_trial
        print("trial_number:", best_trial.number)
        print(f"Value: {best_trial.value}")
        print("Params:")
        for key, value in best_trial.params.items():
            print(f"{key}: {value}")

        output_results(study, model_name, output_type='maximum', output_mode_list=['plotly'])
    
    #For the BERT models
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer)
        
        id2label = {
            0: "O",
            1: "I-TECT",
            2: "B-TECT"
        }
        
        label2id = {
            "O": 0,
            "I-TECT": 1,
            "B-TECT": 2
        }
            
        def objective(trial: optuna.Trial):
            batch_size = trial.suggest_categorical("batch_size", [16, 32])
            learning_rate= trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
            num_train_epochs= trial.suggest_int("num_train_epochs", 1, 7)
            seed= trial.suggest_int("seed", 1, 40)
            warmup_steps= trial.suggest_int("warmup_steps", 0, 1000)
            weight_decay= trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
            
            #Load NER model
            model = AutoModelForTokenClassification.from_pretrained(checkpoint,
                                                                    num_labels=3,
                                                                    id2label=id2label,
                                                                    label2id=label2id,
                                                                    finetuning_task="ner")
            
            #Define training args
            training_arguments = TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_eval_batch_size=batch_size,
                per_device_train_batch_size=batch_size,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                output_dir="hpo",
                save_strategy="no",
                seed=seed)
            
            #Define trainer
            trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            data_collator=data_collator)
            
            trainer.train()
            results = trainer.evaluate()
            objective = results["eval_f1"] + results["eval_recall"] + results["eval_precision"]
            
            return objective 

        #Start HPO
        study = optuna.create_study(sampler=TPESampler(), direction='maximize')  # For accuracy or F1 score
        study.optimize(objective, n_trials=200)
        
        print(f"Best trial for {model_name}:")
        best_trial = study.best_trial
        print("trial_number:", best_trial.number)
        print(f"Value: {best_trial.value}")
        print("Params:")
        for key, value in best_trial.params.items():
            print(f"{key}: {value}")

        output_results(study, model_name, output_type='maximum', output_mode_list=['plotly'])
    
        del tokenizer
        del study
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    models = [("bert-base-uncased","bert_base"),("FacebookAI/roberta-base","roberta_base"), ("FacebookAI/xlm-roberta-large","roberta_large")]

    for model in models:
        run_hpo(model)

    run_hpo("", lstm=True)
        

