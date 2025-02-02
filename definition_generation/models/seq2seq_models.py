import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, T5Tokenizer, T5ForConditionalGeneration
from models.base_model import LLMWrapper
from huggingface_hub import login
import gc
from transformers import BartTokenizerFast, BartForConditionalGeneration

class MyBART(LLMWrapper):

    def initialize(self):
        self.name = "BART (my)"
        self.tokenizer = BartTokenizerFast.from_pretrained("models/files/checkpoint-3864_last")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("models/files/checkpoint-3864_last")
    
    def create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = f"question: What is (are) {tech_term}?"
        
        if sentence != None:
            prompt = f"question: What is (are) {tech_term}? Context: {sentence}"

        return prompt

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if not retriever:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs,
                                       decoder_start_token_id=self.tokenizer.bos_token_id,
                                       num_return_sequences=1,
                                       num_beams=5,
                                       max_length=64,
                                       min_length=8,
                                       early_stopping=True,
                                       temperature=None,
                                       do_sample=True,
                                       top_k=50,
                                       top_p=0.9,
                                       no_repeat_ngram_size=3)
        print(outputs[0])
        answers = [self.tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in outputs[0]]

        return answers
    
    def clear_space(self) -> None:
        del self.tokenizer
        del self.model
        gc.collect()    
        torch.cuda.empty_cache()


class MyT5(LLMWrapper):

    def initialize(self):
        self.name = "T5 (ours)"
        self.tokenizer = T5Tokenizer.from_pretrained("t5_results/checkpoint-20616")
        self.model = T5ForConditionalGeneration.from_pretrained("t5_results/checkpoint-20616")
    
    def create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = f"question: What is (are) {tech_term}?"
        
        if sentence != None:
            prompt = f"question: What is (are) {tech_term}? Context: {sentence}"

        return prompt

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if not retriever:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs,
                                       decoder_start_token_id=self.tokenizer.bos_token_id,
                                       num_return_sequences=1,
                                       num_beams=5,
                                       max_length=64,
                                       min_length=8,
                                       early_stopping=True,
                                       temperature=None,
                                       do_sample=True,
                                       top_k=50,
                                       top_p=0.9,
                                       no_repeat_ngram_size=3)
        print(outputs[0])
        answers = [self.tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in outputs[0]]

        return answers
    
    def clear_space(self) -> None:
        del self.tokenizer
        del self.model
        gc.collect()    
        torch.cuda.empty_cache()
