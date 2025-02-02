"""
This script is used to fine-tune the named entity recognition models. You can fine-tune the BERT models, 
or an LSTM+CRF model.
"""

import torch
import gc
from datasets import ClassLabel
from ner_utils import tokenize_and_align_labels, compute_metrics, load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer

def main(out_path: str, model_path: str) -> None:
    """
    This is the main fine-tuning loop.

    Args:
        - out_path (str): Name of the folder where the checkpoint will be saved
        - model_path (str): Checkpoint of the NER model

    Output:
        - None
    """
    
    #Define tokenizer
    if out_path == "roberta_base" or out_path == "roberta_large":
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    #Load Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    classmap = ClassLabel(num_classes=3, names=['O', 'I-TECT', 'B-TECT'])
    
    #Prepare dataset
    train_dataset, eval_dataset = load_dataset("annotations", tokenizer, classmap, overfit=False)
    
    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})
    
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
    
    #Load model
    model = AutoModelForTokenClassification.from_pretrained(model_path,
                                                            num_labels=3,
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            finetuning_task="ner")
    
    #Define training arguments
    training_args = TrainingArguments(
        output_dir=out_path,
        load_best_model_at_end=True,
        learning_rate=7.21919e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=7.40945e-5,
        warmup_steps = 68,
        seed=31,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("training is now starting...")
    trainer.train()
    
    del tokenizer
    del model
    gc.collect()


if __name__ == '__main__':
    models = [("bert-base-uncased","bert_base"),("FacebookAI/roberta-base","roberta_base"), ("FacebookAI/xlm-roberta-large","roberta_large")]
    main("roberta_large", "FacebookAI/xlm-roberta-large")