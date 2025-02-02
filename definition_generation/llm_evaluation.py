from evaluate import list_evaluation_modules, load
import pandas as pd
import numpy as np
import os
import re

def main(FILE_DIR):
    bertscore = load("bertscore")
    bleu = load("bleu")
    meteor = load("meteor")
    rouge = load("rouge")

    dataset = []

    for file in os.listdir(FILE_DIR):
        filepath = os.path.join(FILE_DIR, file)

        data = pd.read_excel(filepath)

        model_name = filepath.split("\\")[-1].replace("predictions_", "").replace(".xlsx", "")

        references = data['description'].tolist()
        predictions_no_sentence = list(map(clean_prediction, data['prediction no sentence']))
        predictions_w_sentence = list(map(clean_prediction, data['prediction with sentence']))

        results_bertscore_ns = bertscore.compute(predictions=predictions_no_sentence, references=references, model_type="roberta-large-mnli")
        results_bleu_ns = bleu.compute(predictions=predictions_no_sentence, references=references)
        results_meteor_ns = meteor.compute(predictions=predictions_no_sentence, references=references)
        results_rouge_ns = rouge.compute(predictions=predictions_no_sentence, references=references)

        results_bertscore_ws = bertscore.compute(predictions=predictions_w_sentence, references=references, model_type="roberta-large-mnli")
        results_bleu_ws = bleu.compute(predictions=predictions_w_sentence, references=references)
        results_meteor_ws = meteor.compute(predictions=predictions_w_sentence, references=references)
        results_rouge_ws = rouge.compute(predictions=predictions_w_sentence, references=references)

        bert_score_ns = round(np.mean(results_bertscore_ns['f1']), 2)
        bleu_score_ns = round(results_bleu_ns['bleu'], 2)
        meteor_score_ns = round(results_meteor_ns['meteor'], 2)
        r_1_ns = round(results_rouge_ns['rouge1'], 2)
        r_2_ns = round(results_rouge_ns['rouge2'], 2)
        r_l_ns = round(results_rouge_ns['rougeL'], 2)

        bert_score_ws = round(np.mean(results_bertscore_ws['f1']), 2)
        bleu_score_ws = round(results_bleu_ws['bleu'], 2)
        meteor_score_ws = round(results_meteor_ws['meteor'], 2)
        r_1_ws = round(results_rouge_ws['rouge1'], 2)
        r_2_ws = round(results_rouge_ws['rouge2'], 2)
        r_l_ws = round(results_rouge_ws['rougeL'], 2)

        bert_score_ws = 0
        bleu_score_ws = 0
        meteor_score_ws = 0
        r_1_ws = 0
        r_2_ws = 0
        r_l_ws = 0

        data_row = [model_name, r_1_ns, r_2_ns, r_l_ns, meteor_score_ns, bleu_score_ns, bert_score_ns, r_1_ws, r_2_ws, r_l_ws, meteor_score_ws, bleu_score_ws, bert_score_ws]
        dataset.append(data_row)

    r_1_ns, r_2_ns, r_l_ns, meteor_score_ns, bleu_score_ns, bert_score_ns, r_1_ws, r_2_ws, r_l_ws, meteor_score_ws, bleu_score_ws, bert_score_ws = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i, row in enumerate(dataset):
        i += 1
        r_1_ns += row[1]
        r_2_ns += row[2]
        r_l_ns += row[3]
        meteor_score_ns += row[4]
        bleu_score_ns += row[5]
        bert_score_ns += row[6]
        r_1_ws += row[7]
        r_2_ws += row[8]
        r_l_ws += row[9]
        meteor_score_ws += row[10]
        bleu_score_ws += row[11]
        bert_score_ws += row[12]

    avg_r_1_ns = r_1_ns / i
    avg_r_2_ns = r_2_ns / i
    avg_r_l_ns = r_l_ns / i
    avg_meteor_score_ns = meteor_score_ns / i 
    avg_bleu_score_ns = bleu_score_ns / i
    avg_bert_score_ns = bert_score_ns / i 
    avg_r_1_ws = r_1_ws / i 
    avg_r_2_ws = r_2_ws / i 
    avg_r_l_ws = r_l_ws / i 
    avg_meteor_score_ws = meteor_score_ws / i 
    avg_bleu_score_ws = bleu_score_ws / i 
    avg_bert_score_ws = bert_score_ws / i 

    dataset.append(["averages", avg_r_1_ns, avg_r_2_ns, avg_r_l_ns,
    avg_meteor_score_ns, avg_bleu_score_ns, avg_bert_score_ns, avg_r_1_ws, avg_r_2_ws, avg_r_l_ws, avg_meteor_score_ws, avg_bleu_score_ws, avg_bert_score_ws])

    df = pd.DataFrame(dataset, columns=["model_name", "r1_ns", "r2_ns", "rl_ns", "meteor_ns", "bleu_ns", "bert_ns", "r1_ws", "r2_ws", "rl_ws", "meteor_ws", "bleu_ws", "bert_ws" ])

    df.to_excel("llm_evaluation.xlsx", index=False)

if __name__ == '__main__':
    FILE_DIR = r"text_generation\\data\\cleaned_prediction"
    main(FILE_DIR)