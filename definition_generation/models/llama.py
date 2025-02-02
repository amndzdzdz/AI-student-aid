import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from models.base_model import LLMWrapper
from huggingface_hub import login
import gc

access_token = "..."
login(token = access_token)

class Llama33_I(LLMWrapper):

    def initialize(self):
        self.name = "Llama 3.3 70B Instruct"
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        self.pipe = pipeline("text-generation", model=self.model, 
                             tokenizer=self.tokenizer, device_map="auto", model_kwargs={"torch_dtype": torch.bfloat16})

    def create_prompt(self, prompt: str, tech_term: str, sentence = None, retriever=None) -> str:
        if not retriever:
            prompt = prompt.replace("XX_keyword_XX", tech_term)
    
            if sentence != None:
                prompt = prompt.replace("XX_sentence_XX", sentence)

        messages = [
            {"role": "system", "content": "You are a highly knowledgeable scientist with expertise in technical concepts. You provide clear, precise, and concise definitions for technical terms."},
            {"role": "user", "content": prompt}]

        return messages

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if retriever:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence, retriever=retriever)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        else:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
            output = self.pipe(prompt, max_new_tokens=256)

        return output[0]["generated_text"][-1]["content"]
    
    def clear_space(self) -> None:
        del self.model
        del self.tokenizer
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()


class Llama32_3B(LLMWrapper):

    def initialize(self):
        self.name = "Llama 3.2 3B"
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B", 
                             torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=512)

    def create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = prompt.replace("XX_keyword_XX", tech_term)

        if sentence != None:
            prompt = prompt.replace("XX_sentence_XX", sentence)

        return prompt

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if retriever:
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        else:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)

        return output[0]["generated_text"]
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()


class Llama31_70B(LLMWrapper):

    def initialize(self):
        self.name = "Llama 3.1 70B"
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B", load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
        self.pipe = pipeline("text-generation", model=self.model, 
                             tokenizer=self.tokenizer, device_map="auto", max_new_tokens=512)

    def create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = prompt.replace("XX_keyword_XX", tech_term)

        if sentence != None:
            prompt = prompt.replace("XX_sentence_XX", sentence)

        return prompt

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if retriever:
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        else:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)

        return output[0]["generated_text"]
    
    def clear_space(self) -> None:
        del self.model
        del self.tokenizer
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()


class Llama31_8B(LLMWrapper):

    def initialize(self):
        self.name = "Llama 3.1 8B"
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B", 
                             torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=512)

    def create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = prompt.replace("XX_keyword_XX", tech_term)

        if sentence != None:
            prompt = prompt.replace("XX_sentence_XX", sentence)

        return prompt

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if retriever:
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        else:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)

        return output[0]["generated_text"]
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()


class Llama31_I(LLMWrapper):

    def initialize(self):
        self.name = "Llama 3.1 8B Instruct"
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct",
                             torch_dtype=torch.bfloat16,  device_map="auto", max_new_tokens=256)

    def create_prompt(self, prompt: str, tech_term: str, sentence = None, retriever=None) -> str:
        if not retriever:
            prompt = prompt.replace("XX_keyword_XX", tech_term)
    
            if sentence != None:
                prompt = prompt.replace("XX_sentence_XX", sentence)

        messages = [
        {"role": "system", 
         "content": "You are an expert assistant specialized in technical terms. Provide clear, accurate and brief definitions for technical terms."},
        {"role": "user", 
         "content": prompt}]
        
        return messages

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if retriever:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence, retriever=retriever)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        else:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)

        return output[0]["generated_text"][-1]["content"]
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()


class Llama32_I(LLMWrapper):

    def initialize(self):
        self.name = "Llama 3.2 3B Instruct"
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", 
                             torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=256)

    def create_prompt(self, prompt: str, tech_term: str, sentence = None, retriever=None) -> str:
        if not retriever:
            prompt = prompt.replace("XX_keyword_XX", tech_term)
    
            if sentence != None:
                prompt = prompt.replace("XX_sentence_XX", sentence)

        messages = [
        {"role": "system", 
         "content": "You are an expert assistant specialized in technical terms. Provide clear, accurate and brief definitions for technical terms."},
        {"role": "user", 
         "content": prompt}]
        
        return messages

    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        if retriever:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence, retriever=retriever)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        else:
            prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
            output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)

        return output[0]["generated_text"][-1]['content']
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()