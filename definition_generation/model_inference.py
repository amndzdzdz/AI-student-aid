"""
This script performs inference using various language models to generate predictions 
based on provided data and prompts. It supports both standard inference, RAG (retrieval-augmented generation) and few-shot prompting.
"""

from models.llama import Llama33_I, Llama32_3B, Llama31_8B, Llama31_70B, Llama31_I, Llama32_I
from models.phi import Phi
from models.ministral import Ministral8B, Mixtral
from models.bart import BARTSciDef, MyBARTSciDef
from models.galactica import Galactica
from models.gpt import GPT2_L
import pandas as pd
from tqdm import tqdm
from utils import index_data, generate_rag_response
import random

def run_inference(num_shots, rag=False):
    """
    This is the main inference loop, it reads the inference data and the prompt and loops over every LLM, 
    generating definition generations.

    Args:
        - num_shots (int): The number of few-shot samples
        - rag (bool): Boolean, if true use RAG

    Output:
        - none    
    """

    DATA_PATH = "data/dataset/dataset_with_sentences.xlsx"
    PROMPTS_PATH = "data/prompts/prompts.xlsx"
    
    data = pd.read_excel(DATA_PATH)
    prompts = pd.read_excel(PROMPTS_PATH)
    
    models = [Mixtral(), Llama31_I()] #add more models 

    data_df = pd.read_csv("SciWiGlossaries_train_OG.csv")
    template = "Explain the term XX_keyword_XX briefly. XX_sentence_XX"

    #setup vector store when using RAG
    if rag:
        retriever = index_data("data/xtended_rag_no_ind.csv", k_retrievals=3)
    
    for model in models:
        
        #initialize model
        model.initialize()
        
        for row_index, row in tqdm(data.iterrows(), "iterating over data..."):

            #Additional few-shot prompting setup
            if not rag: 
                few_shot_prompts = []
                for i in range(num_shots):
                    random_int = random.randint(0, len(data_df) - 1)
                    question = data_df.iloc[random_int]['question']
                    answer = data_df.iloc[random_int]['answer'].capitalize()
                    answer = "Answer: " + answer
                    term = question.replace("What is (are) ", "")[:-1]
                    prompt = template.replace("XX_keyword_XX", term)
                    prompt = prompt.replace("XX_sentence_XX", answer)
                    prompt = "Instruction: " + prompt
                    few_shot_prompts.append(prompt)
                    
                few_shot_prompts = "\n".join(few_shot_prompts)
                
            term, _, sentence = row['keyword'], row['description'], row['sentence']

            prompt_no_sentence = prompts['prompts'][0]
            prompt_w_sentence = prompts['prompts'][1]

            # #prompt_no_sentence = "\n".join([few_shot_prompts, prompt_no_sentence])
            # print(prompt_no_sentence)
            # print(type(prompt_no_sentence))
            
            if rag:
                prediction_no_sentence = generate_rag_response(term, sentence, model, retriever)
                prediction_w_sentence = ""
            else:
                prediction_no_sentence = model.predict(prompt_no_sentence, term)
                prediction_w_sentence = model.predict(prompt_w_sentence, term, sentence)

            try:
                data.insert(len(data.columns), "prediction no sentence", ['nan' for i in range(len(data))])
                data.insert(len(data.columns), "prediction with sentence", ['nan' for i in range(len(data))])
            except:
                print("")
    
            data.at[row_index, "prediction no sentence"] = prediction_no_sentence
            data.at[row_index, "prediction with sentence"] = prediction_w_sentence
    
        data.to_excel(f"data/predictions/predictionss_{model.name}.xlsx")
    
        model.clear_space()

if __name__ == '__main__':
    run_inference(num_shots=3, rag=True)