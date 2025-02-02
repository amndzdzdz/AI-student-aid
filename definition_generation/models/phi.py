import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from models.base_model import LLMWrapper
from huggingface_hub import login
import gc

class Phi(LLMWrapper):

    def initialize(self):
        self.name = "Phi 3 mini Instruct"
        self.pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", 
                             torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=512, trust_remote_code=True)

    def create_prompt(self, prompt: str, tech_term: str, sentence = None, retriever=None) -> str:
        if not retriever:
            prompt = prompt.replace("XX_keyword_XX", tech_term)
    
            if sentence != None:
                prompt = prompt.replace("XX_sentence_XX", sentence)

        messages = [
            {"role": "system", 
             "content": "You are an expert assistant specialized in technical terms. Provide clear, accurate and brief definitions for technical terms."},
            {"role": "user", "content": prompt}
        ]
        
        return messages

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if retriever:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence, retriever=retriever)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        else:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)

        return output[0]['generated_text'][-1]['content']
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()
