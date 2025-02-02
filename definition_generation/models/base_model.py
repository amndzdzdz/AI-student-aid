from abc import ABC, abstractmethod

class LLMWrapper(ABC):
    """
    Parent class that the LLMs will inherit from. Defines the needed functions.
    """

    @abstractmethod
    def initialize(self):
        """
        This function initializes the model name and the model itself (could be the model + tokenizer, only the pipeline, 
        or a combination of both, depending on the model).

        Args:
            /
        
        Output:
            /
        """
        pass

    @abstractmethod
    def create_prompt(self, prompt: str, tech_term: str, sentence = None, retriever=None):
        """
        This function receives a prompt template, the technical term, the sentence the term was found in, and a boolean
        and generates the prompt that the model will receive.

        Args:
            - prompt (str): The prompt template
            - tech_term (str): The technical term
            - sentence (str): The sentence that the term was found in
            - retriever (bool): Boolean, if true, use a different string replacement, suitable for RAG
        
        Output:
            - messages (str / list): Returns a string or a list (base-models vs. instruction tuned models) that contains the prompt
        """
        pass

    @abstractmethod
    def predict(self, prompt:str, tech_term: str, sentence=None, retriever=None):
        """
        This function receives the prompt template, the technical term, the sentence it was found in,
        and a boolean. The 'create_prompt' method is called and the created prompt is fed to the intialized
        model. The function outputs the prediction.

        Args:
            - prompt (str): The prompt template
            - tech_term (str): The technical term
            - sentence (str): The sentence that the term was found in
            - retriever (bool): Boolean, if true, use a different string replacement, suitable for RAG
        
        Output:
            - (str): The LLM's prediction
        """
        pass

    @abstractmethod
    def clear_space(self):
        """
        This function deletes the initialized model, empties the cuda cache, and collects the garbage,
        in order to free the GPU memory

        Args:
            /
        
        Output:
            /
        """
        pass