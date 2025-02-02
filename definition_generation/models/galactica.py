import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from models.base_model import LLMWrapper
from huggingface_hub import login
import gc

access_token = "..."
login(token = access_token)

class Galactica(LLMWrapper):

    def initialize(self):
        self.name = "Galactica 7B"
        self.pipe = pipeline(model="facebook/galactica-6.7b", torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=512)

    def __create_prompt(self, prompt: str, tech_term: str, sentence = None, retriever=None) -> str:
        if not retriever:
            prompt = prompt.replace("XX_keyword_XX", tech_term)
    
            if sentence != None:
                prompt = prompt.replace("XX_sentence_XX", sentence)
        return prompt + "[START_REF]"

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if retriever:
            prompt = self.__create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence, retriever=retriever)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        else:
            prompt = self.__create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)

        output = output[0]["generated_text"]

        return output
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()