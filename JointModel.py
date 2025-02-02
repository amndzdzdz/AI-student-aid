"""
This script contains the class for the JointModel that consists of a named entity recognition model and a text generation model
"""

from definition_generation.models.phi import Phi
from transformers import AutoTokenizer
from transformers import pipeline
import re

class JointModel():
    """
    The JointModel's main jobs are:
            1. It extracts technical terms through the function 'extract_terms'
            2. It generates term definitions through the function 'generate_definitions'
    """
    
    def __init__(self, ner_model_checkpoint: str, llm_model) -> None:
        """
        This function initializes the JointModel. 

        Args:
            - ner_model_checkpoint (str): The checkpoint of the NER model
            - llm_model: An LLM model object

        Output:
            - None
        """
        self.definitionGenerator = llm_model
        self.ner_model_checkpoint = ner_model_checkpoint
        self.template = "What is (are) {term}? Context: {context} Define the term briefly."
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model_checkpoint)

        self.nermodel = pipeline("ner", 
                                 model=ner_model_checkpoint, 
                                 tokenizer=self.tokenizer, 
                                 device='cuda')

    def extract_terms(self, text: str) -> list:
        """
        This method extracts for a given text all the technical terms which occur in it and returns them as a list of dictionaries
        i. e.  [{'start': 29, 'end': 32, 'word': 'cpu', 'score': 0.9987238},
                {'start': 66, 'end': 70, 'word': 'cuda', 'score': 0.9958228}]

        Args:
            -text (str): The input text from which the technical terms are to be extracted

        Return:
            -named_entities (list[dict]): A list of dictionaries, where each dictionary contains a technical term, the start & end index and the confidence score
        """
        results = self.nermodel(text)
        formatted_results = self.__format_entity_predictions(results)
        named_entities = self.__combine_entities(formatted_results, text)
        return named_entities

    def generate_definitions(self, term_dicts: list, sentence: str) -> list:
        """
        This method takes in the technical terms dictionaries and the text they were found in and generates descriptions for each term.
        The output is a list of dictionaries with the technical term and the corresponding definition.
        i. e.  [{'term': 'cpu', 'definition': 'A computer processing unit ( CPU ) designed to accelerate computation.'},
                {'term': 'cuda','definition': '   A term sometimes used to refer to the structure of a plant . '}]

        Args:
            -term_dicts (list[dict]): The list of technical terms dictionaries
            -sentence (str): The text the technical terms occured in

        Return:
            -term_def_pairs (list[dict]): A list of dictionaries which contain the techincal term and the definition
        """
        term_def_pairs = []
        for term_dict in term_dicts:
            term = term_dict['word']
            definition = self.definitionGenerator.predict(self.template, term)
            term_def_pairs.append({"term": term, "definition": definition})

        return term_def_pairs

    def __combine_entities(self, entities: list, text: str) -> list:
        """
        This method combines consecutive entities labeled as 'B-TECT' and 'I-TECT' into a single entity, adjusting their positions to match full tokens.
        The method ensures that the final entities are aligned with word boundaries and returns the adjusted list of entities.

        Args:
            - entities (list[dict]): A list of entities, where each entity contains attributes such as 'start', 'end', 'word', 'score', and 'entity'.
            - text (str): The original text from which the entities were extracted.

        Returns:
            - adjusted_entities (list[dict]): A list of combined and adjusted entities with corrected token boundaries.
        """
        combined_entities = []
        temp_entity = None

        # Combine B-TECT and I-TECT entities
        for entity in entities:
            if entity['entity'] == 'B-TECT':
                if temp_entity:
                    combined_entities.append(temp_entity)
                temp_entity = {
                    'start': entity['start'],
                    'end': entity['end'],
                    'word': entity['word'],
                    'score': entity['score']
                }
            elif entity['entity'] == 'I-TECT' and temp_entity:
                temp_entity['end'] = entity['end']
                temp_entity['word'] += f" {entity['word']}"
                temp_entity['score'] = (temp_entity['score'] + entity['score']) / 2
            else:
                if temp_entity:
                    combined_entities.append(temp_entity)
                    temp_entity = None
                combined_entities.append(entity)

        if temp_entity:
            combined_entities.append(temp_entity)

        # Helper function to extend entity boundaries to full tokens
        def extend_to_full_token(text, start, end):
            while start > 0 and text[start - 1].isalnum():
                start -= 1
            while end < len(text) and text[end].isalnum():
                end += 1
            return start, end

        # Adjust entity boundaries to full tokens
        adjusted_entities = []
        for entity in combined_entities:
            start, end = extend_to_full_token(text, entity['start'], entity['end'])
            token = text[start:end]

            if token and token[-1] in ["!", "?", ",", ".", ";"]:
                token = token[:-1]

            adjusted_entities.append({
                'start': start,
                'end': end,
                'word': token,
                'score': entity['score']
            })

        return adjusted_entities

    def __format_entity_predictions(self, results: list) -> list:
        """
        Formats the entity predictions by merging word pieces (subwords) if applicable. 
        The method is designed to handle models like BERT that may split words into subword tokens.

        Args:
            - results (list[dict]): A list of predicted entities with attributes such as 'start', 'word', 'entity', 'score', and 'index'.

        Returns:
            - formatted_results (list[dict]): A list of formatted entities with merged word pieces.
        """
        if "roberta" in self.ner_model_checkpoint:
            return None

        formatted_results = []
        for result in results:
            end = result["start"] + len(result["word"].replace("##", ""))

            if result["word"].startswith("##"):
                # Merge with the previous entity if it is a subword
                formatted_results[-1]["end"] = end
                formatted_results[-1]["word"] += result["word"].replace("##", "")
            else:
                # Add a new entity to the results
                formatted_results.append({
                    'start': result["start"],
                    'end': end,
                    'entity': result["entity"],
                    'index': result["index"],
                    'score': result["score"],
                    'word': result["word"]
                })

        return formatted_results

    def __clear_space(self) -> None:
        """
        Clears model resources from memory to free up space.
        This method deletes the model, tokenizer, and definition generator, and empties the CUDA cache to reduce memory usage.

        Args:
            - None

        Returns:
            - None
        """
        del self.nermodel
        del self.tokenizer
        del self.definitionGenerator
        gc.collect()
        torch.cuda.empty_cache()
