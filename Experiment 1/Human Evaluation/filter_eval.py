import os
import gradio as gr
import json
import pandas as pd
from random import shuffle
import itertools
import numpy as np
import math


def clean_response(response):
    if isinstance(response, float) and math.isnan(response):
        return None

    else:
        response = response.strip()
        response = response.replace('speaker a:', "").replace('speaker b:', "").replace('a:', "").replace('b:', "").replace('A:', "").replace('B:', "")
        if response[0] == "'" or response[0] == '"': response = response[1:]

        n = len(response) -1
        if response[n] == "'" or response[n] == '"': response = response[:n]


    return response

def load_conversations(path, number_samples=None):
    data = pd.read_csv(path, encoding="UTF-8")
    
    if number_samples is not None:
        data = data.iloc[15:number_samples]
    candidates_columns = [x for x in data.columns if x != "context"]

    contexts = data['context'].tolist()
    responses = []

    for i, row in data.iterrows():
        candidates = [clean_response(row[col]) for col in candidates_columns if clean_response(row[col]) is not None]
        unique_candidates = []
        for candidate in candidates:
            if candidate.lower() not in [x.lower() for x in unique_candidates]:
                unique_candidates.append(candidate)
        responses.append(unique_candidates)

    return {'context': contexts, 'candidates': responses}
    

from ast import literal_eval

def prep_batch(batch):
    cols = [x for x in batch.columns if x not in ['id', 'nonsense']]
    for col in cols:
        batch[col] = batch[col].apply(lambda x: literal_eval(x) if "[" in x else x)
    
    batch['id'] = batch['id'].apply(lambda x: [int(a) for a in x.replace('[', "").replace(']', "").split(', ')])
    return batch.to_dict()


def get_sets(batch, id):
    assert(len(batch['candidates'][id])==len(batch['elimination'][id]))
    ok_candidates = [rep for i, rep in enumerate((batch['candidates'][id])) if (batch['elimination'][id])[i] == 0]
    ko_candidates = [rep for i, rep in enumerate((batch['candidates'][id])) if (batch['elimination'][id])[i] == 'X']
    to_rank = [rep for i, rep in enumerate((batch['candidates'][id])) if (batch['rank'][id])[i] != 'X']

    return gr.State(ok_candidates), gr.State(ko_candidates), gr.State(to_rank)


def elimination(batch, slider_id, button_value, save_path):
    if isinstance(batch, dict):
        batch = gr.State(batch)

    if isinstance(save_path, str):
        save_path = gr.State(save_path)
    batch = batch.value

    #print(button_value)
    assert(len(batch['candidates'][slider_id]) == len(batch['elimination'][slider_id]))
    for idx in range(len(batch['candidates'][slider_id])):
        if batch['candidates'][slider_id][idx] in button_value:
            batch['elimination'][slider_id][idx] = 'X'
            batch['rank'][slider_id][idx] = 'X'
        
        else:
            batch['elimination'][slider_id][idx] = 0
            if batch['rank'][slider_id][idx] == "X":
                batch['rank'][slider_id][idx] = 0
    
    #print(button_value)
    #print(batch['elimination'][slider_id])
    #print('wtf elim')
    pd.DataFrame(batch).to_csv(save_path.value, encoding='UTF_8', index= False)
    ok_candidates, ko_candidates, to_rank = get_sets(batch, slider_id)

    str_ko = display_responses(batch, slider_id, ko_candidates)
    str_ok = display_responses(batch, slider_id, ok_candidates)

    #print(ko_candidates.value)
    return batch, button_value, ok_candidates.value, ko_candidates.value, to_rank.value, str_ok.value, str_ko.value


def display_responses(batch, slider_id, responses):
    if isinstance(batch, dict):
        batch = gr.State(batch)

    if isinstance(responses, list):
        responses = gr.State(responses)

    #print(responses.value)
    string_response = [""] *len(batch.value['candidates'][slider_id])
    for resp in responses.value:
        curr_id = batch.value['candidates'][slider_id].index(resp)
        #print(batch.value['candidates'][slider_id], resp)
        #print(curr_id, batch.value['id'][slider_id])
        curr_id = batch.value['id'][slider_id][curr_id]
        #print(curr_id)
        #ids.append(curr_id)
    
        string_response[curr_id-1] = str(curr_id) + ": " + resp
    
    #print(string_response)
    string_response = '\n'.join([x for x in string_response if x!= ""])
    
    #print(string_response)
    string_response = gr.State(string_response)
    return string_response


def rerank(batch, slider_id, resp_1, resp_2, resp_3, save_path):
    if isinstance(batch, dict):
        batch = gr.State(batch)

    
    if isinstance(save_path, str):
        save_path = gr.State(save_path)
    
    batch = batch.value

    print(len(batch['candidates']))
    assert(len(batch['candidates'][slider_id]) == len(batch['rank'][slider_id]))

    order_list= ["", "", ""]

    for idx in range(len(batch['candidates'][slider_id])):
        if batch['id'][slider_id][idx] == int(resp_1):
            if batch['elimination'][slider_id][idx] == 'X':
                order_list[0] = "Cette réponse fait partie des réponses éliminées. Je ne peux pas te forcer à mettre une autre réponse parce que je galère avec cette plateforme, mais si possible, pourrais-tu choisir une réponse valide? Cimer."

            else:
                batch['rank'][slider_id][idx] = 1
                order_list[0] = str(int(resp_1)) + ": " + batch['candidates'][slider_id][idx]
        
        elif batch['id'][slider_id][idx] == int(resp_2):
            if batch['elimination'][slider_id][idx] == 'X':
                order_list[1] = "Cette réponse fait partie des réponses éliminées. Je ne peux pas te forcer à mettre une autre réponse parce que je galère avec cette plateforme, mais si possible, pourrais-tu choisir une réponse valide? Cimer."

            else:
                batch['rank'][slider_id][idx] = 2
                order_list[1] = str(int(resp_2)) + ": " + batch['candidates'][slider_id][idx]

        
        elif batch['id'][slider_id][idx] == int(resp_3):
            if batch['elimination'][slider_id][idx] == 'X':
                order_list[2] = "Cette réponse fait partie des réponses éliminées. Je ne peux pas te forcer à mettre une autre réponse parce que je galère avec cette plateforme, mais si possible, pourrais-tu choisir une réponse valide? Cimer."

            else:
                batch['rank'][slider_id][idx] = 3
                order_list[2] = str(int(resp_3)) + ": " + batch['candidates'][slider_id][idx]


        elif batch['rank'][slider_id][idx] in [1,2,3]:
            if batch['elimination'][slider_id][idx] == 'X':
                batch['rank'][slider_id][idx] = 'X'
            
            else:
                batch['rank'][slider_id][idx] = 0

    pd.DataFrame(batch).to_csv(save_path.value, encoding='UTF_8', index= False)
    order_str = '\n\n'.join([x for x in order_list if x!= ""])

    return batch, order_str

def print_context(context):
    context = [x for x in context.replace('<sp1>','<SEP>').replace('<sp2>','<SEP>').split('<SEP>') if len(x) > 2]
    context = "SPEAKER A: " + context[0]+ "\nSPEAKER B: " + context[1]+"\nSPEAKER A: " + context [2]
    return context

def display(batch, sample_id_slider, disp_candidates=None):
    if isinstance(batch, dict):
        batch = gr.State(batch)


    sample_id = sample_id_slider
    context = batch.value['context'][sample_id]
    context = gr.State(print_context(context))
    candidates = gr.State(batch.value['candidates'][sample_id])
    ok_candidates, ko_candidates, to_rank = get_sets(batch.value, sample_id)

    if disp_candidates is None:
        disp_candidates = gr.CheckboxGroup(choices = candidates.value, value = ko_candidates, label = 'Réponses Candidates à Filtrer', info="Filtrer les réponses par cohérence: séléctionnez les réponses qui ne sont pas pertinente pour les éliminer.")

    else:
        disp_candidates = update_choices(disp_candidates, candidates.value, ok_candidates)

    order_str, str_ko, str_ok = "", "", ""
    resp_1, resp_2, resp_3 = 0, 0, 0

    return context.value, sample_id, shuffle(candidates.value), ok_candidates.value, ko_candidates.value, to_rank.value, disp_candidates, order_str, str_ko, str_ok, resp_1, resp_2, resp_3
    #return context


def update_choices(disp_candidates, candidates, ko_candidates):
    return gr.CheckboxGroup.update(choices=candidates, value=ko_candidates)

import webbrowser


def next(batch, sample_id_slider):
    if isinstance(batch, dict):
        batch = gr.State(batch)

    disp_candidates = None
    if sample_id_slider < (len(batch.value['context']) - 1):
        sample_id_slider = sample_id_slider + 1

        candidates = gr.State(batch.value['candidates'][sample_id_slider])
        ok_candidates, ko_candidates, to_rank = get_sets(batch.value, sample_id_slider)
        disp_candidates = gr.CheckboxGroup(choices = candidates.value, value = ko_candidates, label = 'Réponses Candidates à Filtrer', info="Filtrer les réponses par cohérence: séléctionnez les réponses qui ne sont pas pertinente pour les éliminer.")

    return display(batch, sample_id_slider, disp_candidates)

path = '/data/Amira_evaluation_data.csv'
conversations = load_conversations(path)


with gr.Blocks() as demo:
    with gr.Row():
        save_path = gr.State(path)
        batch = pd.read_csv(path, encoding='UTF-8')
        batch = gr.State(prep_batch(batch))
        total_examples = gr.State(len(batch.value['candidates']))

    with gr.Row():
        with gr.Row():
            sample_id = 0
            sample_id_slider = gr.Slider(minimum =0, maximum = total_examples.value - 1, value=0, label="Exemple (total: "+ str(total_examples.value) +")", step = 1)
        
            sample_id = sample_id_slider.value
        
        with gr.Row():
            context = gr.Textbox(print_context(batch.value['context'][sample_id]), lines = 3, interactive=False, label="Contexte")
            
            candidates = gr.State(batch.value['candidates'][sample_id])

            ok_candidates, ko_candidates, to_rank = get_sets(batch.value, sample_id)

    with gr.Row():
        disp_candidates = gr.CheckboxGroup(candidates.value, label = 'Réponses Candidates à Filtrer', info="Filtrer les réponses par cohérence: séléctionnez les réponses qui ne sont pas pertinente pour les éliminer.")
        
    with gr.Row(): 
        filt = gr.Button(value="Filtrer les réponses pertinentes")

    with gr.Row():
        with gr.Column():
            str_ok = gr.Textbox('', lines = 10, interactive=False, label = 'Réponses Pertinentes')

        with gr.Column():
            str_ko = gr.Textbox('', lines = 10, interactive=False, label = 'Réponses Éliminées')
        
    filt.click(fn=elimination, inputs=[batch, sample_id_slider, disp_candidates, save_path], outputs=[batch, disp_candidates, ok_candidates, ko_candidates, to_rank, str_ok, str_ko])

    with gr.Row():
        gr.Textbox(value = "Sélectionner le top-3 des meilleures réponses (les critères sont la cohérence logique et la spécificité). Indiquer leurs ID (le numéro affiché avant la réponse dans les blocs ci-dessus).", interactive=False, label= "Réponses à Ordonner")
    
    with gr.Row():
        order_str = gr.State("")
        with gr.Column():
            resp_1 = gr.Number(value = None, label="ID de la meilleure réponse", minimum=0, maximum = (len(candidates.value)+1))
            resp_2 = gr.Number(value = None, label="ID de la 2ème meilleure réponse", minimum=0, maximum = (len(candidates.value)+1))
            resp_3 = gr.Number(value = None, label="ID de la 3ème meilleure réponse", minimum=0, maximum = (len(candidates.value)+1))

        rank_save = gr.Button(value="Enregistrer l'ordre")

        with gr.Column():
            order_str = gr.Textbox(order_str.value, lines = 3, interactive=False, label = 'Top-3 des réponses')
        
        rank_save.click(fn=rerank, inputs=[batch, sample_id_slider, resp_1, resp_2, resp_3, save_path], outputs=[batch, order_str])

    with gr.Row():
        next_context = gr.Button(value="Contexte Suivant")
        next_context.click(fn=next, inputs=[batch, sample_id_slider], outputs=[context, sample_id_slider, candidates, ok_candidates, ko_candidates, to_rank])

    sample_id_slider.change(fn=display, inputs=[batch, sample_id_slider, disp_candidates], outputs=[context, sample_id_slider, candidates, ok_candidates, ko_candidates, to_rank, disp_candidates, order_str, str_ko, str_ok, resp_1, resp_2, resp_3])

demo.launch(share=True, server_port=7864)