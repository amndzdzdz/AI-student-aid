"""
This script is for inference purposes. You can manually select which sample from the eval_dataset you want to run the inference on.
"""

import os
from datasets import ClassLabel
from ner_utils import load_dataset, save_inference
from transformers import AutoTokenizer, pipeline
from flair.models import SequenceTagger
from flair.data import Sentence
import json

def run(outpath: str, checkpoint_name: str, lstm=False) -> None:
    """
    This function does the inference run. You have to manually slect the elements that you want to run the inference
    on from the eval dataset. The inference results are saved locally in json.

    Args:
        - outpath (str): Path where the results will be saved
        - checkpoint_name (str): Checkpoint of the model that you want to use 
        - lstm (bool): If true, use the LSTM model and not the BERT models

    Output:
        - None
    """
    classmap = ClassLabel(num_classes=3, names=['O', 'I-TECT', 'B-TECT'])
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(outpath, checkpoint_name))
    _, eval_dataset = load_dataset("annotations", tokenizer, classmap, overfit=False)
    
    if lstm:

        #Load the model
        model = SequenceTagger.load(checkpoint_name)

        text1 = eval_dataset['tokens'][70]
        labels1 = eval_dataset['ner_tags'][70]
        inf1 = ' '.join(text1)
        inf1 = Sentence(inf1)
        
        text2 = eval_dataset['tokens'][71]
        labels2 = eval_dataset['ner_tags'][71]
        inf2 = ' '.join(text2)
        inf2 = Sentence(inf2)
        
        text3 = eval_dataset['tokens'][72]
        labels3 = eval_dataset['ner_tags'][72]
        inf3 = ' '.join(text3)
        inf3 = Sentence(inf3)
        
        text4 = eval_dataset['tokens'][73]
        labels4 = eval_dataset['ner_tags'][73]
        inf4 = ' '.join(text4)
        inf4 = Sentence(inf4)
        
        model.predict(inf1)
        model.predict(inf2)
        model.predict(inf3)
        model.predict(inf4)

        inf1.to_tagged_string()
        inf2.to_tagged_string()
        inf3.to_tagged_string()
        inf4.to_tagged_string()

        save_inference(text1, inf1, labels1, os.path.join("lstm_train", "inference1.json"), lstm=True)
        save_inference(text2, inf2, labels2, os.path.join("lstm_train", "inference2.json"), lstm=True)
        save_inference(text3, inf3, labels3, os.path.join("lstm_train", "inference3.json"), lstm=True)
        save_inference(text4, inf4, labels4, os.path.join("lstm_train", "inference4.json"), lstm=True)

    else:
        text1 = eval_dataset['tokens'][22]
        labels1 = eval_dataset['ner_tags'][22]
        inf1 = ' '.join(text1)
        
        text2 = eval_dataset['tokens'][13]
        labels2 = eval_dataset['ner_tags'][13]
        inf2 = ' '.join(text2)
        
        text3 = eval_dataset['tokens'][47]
        labels3 = eval_dataset['ner_tags'][47]
        inf3 = ' '.join(text3)
        
        text4 = eval_dataset['tokens'][19]
        labels4 = eval_dataset['ner_tags'][19]
        inf4 = ' '.join(text4)

        #Load the model
        classifier = pipeline("ner", model=os.path.join(outpath, checkpoint_name), tokenizer=tokenizer, device='cuda')
        
        result1 = classifier(inf1)
        result2 = classifier(inf2)
        result3 = classifier(inf3)
        result4 = classifier(inf4)
        
        save_inference(text1, result1, labels1, os.path.join(outpath, "inference1.json"))
        save_inference(text2, result2, labels2, os.path.join(outpath, "inference2.json"))
        save_inference(text3, result3, labels3, os.path.join(outpath, "inference3.json"))
        save_inference(text4, result4, labels4, os.path.join(outpath, "inference4.json"))

if __name__ == '__main__':
    run('', 'roberta_base_gen_best/checkpoint-343', lstm=False)

