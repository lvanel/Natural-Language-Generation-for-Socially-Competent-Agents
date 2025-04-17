import os
import gradio as gr
import json
import pandas as pd
from random import shuffle
import itertools
import numpy as np
import math
from ast import literal_eval


def prep_batch(batch):
    cols_str = ['context', 'responses', 'annotated_strategy', 'emotion']
    for col in cols_str:
        batch[col] = batch[col].apply(lambda x: literal_eval(x) if "[" in x else x)
 
    cols_int = [c for c in batch.columns if c not in cols_str]

    for col in cols_int:
        batch[col] = batch[col].apply(lambda x: [int(a) for a in x.replace('[', "").replace(']', "").split(', ')])

    return batch


path = '/data/Amira_annotation_data.csv'

def print_context(context):
    context = [x for x in context.replace('<sp1>','<SEP>').replace('<sp2>','<SEP>').split('<SEP>') if len(x) > 2]
    context = "SPEAKER A: " + context[0]+ "\nSPEAKER B: " + context[1]+"\nSPEAKER A: " + context [2]
    return context

def display(batch, sample_id_slider):
    if isinstance(batch, pd.DataFrame):
        batch = gr.State(batch)

    context = batch.value['context'][sample_id_slider]
    context = gr.State(print_context(context))
    curr_candidate = batch.value['responses'][sample_id_slider]
    curr_candidate_annot = batch.value['responses'][sample_id_slider]
    usefulness = 2
    fluency = 2
    emot = []
    emot_adequate = 2
    strat_adequate = 2
    role_consistency = 2
    
    return context.value, sample_id_slider, curr_candidate, curr_candidate_annot, usefulness, fluency, emot, emot_adequate, strat_adequate, role_consistency


def next(batch, sample_id_slider):
    if isinstance(batch, pd.DataFrame):
        batch = gr.State(batch)

    if sample_id_slider < (len(batch.value['context']) - 1):
        return display(batch, (sample_id_slider + 1))

    else:
        return display(batch, sample_id_slider)


def save_annot(batch, sample_id_slider, save_path, curr_candidate_annot, usefulness, fluency, emot, emot_adequate, strat_adequate, role_consistency):
    if isinstance(batch, pd.DataFrame):
        batch = gr.State(batch)

    if isinstance(save_path, str):
        save_path = gr.State(save_path)
    
    #print(sample_id_slider, usefulness, fluency, emot, emot_adequate, strat_adequate, role_consistency)
    batch.value['annotated_strategy'][sample_id_slider] = curr_candidate_annot
    batch.value['usefulness'][sample_id_slider] = usefulness
    batch.value['fluency'][sample_id_slider] = fluency
    batch.value['emotion'][sample_id_slider] = emot
    batch.value['emotion_adequate'][sample_id_slider] = emot_adequate
    batch.value['strategy_adequate'][sample_id_slider] = strat_adequate
    batch.value['role_consistency'][sample_id_slider] = role_consistency

    df = batch.value
    df.to_csv(save_path.value, encoding='UTF_8', index= False)

    #return batch
    return next(batch, sample_id_slider)


with gr.Blocks() as demo:
    with gr.Row():
        save_path = gr.State(path)
        batch = pd.read_csv(path, encoding='UTF-8')
        batch = gr.State(prep_batch(batch))
        total_examples = gr.State(len(batch.value['responses']))

        sample_id_slider = gr.Slider(minimum =0, maximum = total_examples.value - 1, value=0, label="Exemple (total: "+ str(total_examples.value) +")", step = 1)
    
        sample_id = sample_id_slider.value
    with gr.Row():

        context = gr.Textbox(print_context(batch.value['context'][sample_id]), lines = 3, interactive=False, label="Contexte")
        emotions = ['happiness', 'disgust', 'anger', "surprise", "sadness", "fear"]
        curr_candidate = gr.Textbox(batch.value['responses'][sample_id], lines = 3, interactive=False, label="Réponse")

    with gr.Accordion('Stratégie de Dialogue'):
        curr_candidate_annot = gr.Textbox(batch.value['responses'][sample_id], interactive=True, info="Annoter en stratégie de dialogue.", label="Stratégies de Dialogue")


    with gr.Row():
        usefulness = gr.Slider(minimum =1, maximum = 3, value= 2, label="Intérêt de la réponse", info = "Est-ce que cette réponse est intéressante ? Apporte quelque chose à la conversation pour l’interlocuteur ?", step = 1)
        fluency = gr.Slider(minimum =1, maximum = 3, value=2, label="Fluidité", info = "Est-ce que le bon langage /vocabulaire est utilisé ? Est-ce que le français/l’anglais est correct ? grammaticalement… ", step = 1)

    with gr.Row():
        emot = gr.CheckboxGroup(emotions,label = 'Émotions', info="Une émotion est-elle exprimée dans la réponse ? Sélectionner les émotions présentes dans la réponse.")
        emot_adequate = gr.Slider(minimum =1, maximum = 3, value=2, label = 'Ton Émotionnel de la réponse', info="Est-ce que le ton de la réponse vous semble approprié par rapport au contexte ? ", step = 1)

    with gr.Row():
        strat_adequate = gr.Slider(minimum =1, maximum = 3, value=2, label = 'Cohérence des Stratégies de dialogue', info="Est-ce que les stratégies de dialogues sont appropriées (au contexte de la conversation et de l’extrait d’historique) ? ", step = 1)
        role_consistency = gr.Slider(minimum =1, maximum = 3, value=2, label = 'Respect du rôle locuteur', info="Est-ce que le rôle du locuteur est bien respecté ?", step = 1)

    with gr.Row():
        save_annot_button = gr.Button(value="Sauvegarder l'annotation")
        save_annot_button.click(fn=save_annot, inputs=[batch, sample_id_slider, save_path, curr_candidate_annot, usefulness, fluency, emot, emot_adequate, strat_adequate, role_consistency], outputs=[context, sample_id_slider, curr_candidate, curr_candidate_annot, usefulness, fluency, emot, emot_adequate, strat_adequate, role_consistency])

    with gr.Accordion("Cheat Sheet"):
        gr.Textbox("Stratégies de Dialogue:\n<I></I>: The Inform class contains all statements and questions by which the speaker is providing information.\n<Q></Q>: The Questions class is labeled when the speaker wants to know something and seeks for some information.\n<D></D>: The Directives class contains dialog acts like request, instruct, suggest and accept/reject offer. \n<C></C>:The Commissive class is about accept/reject request or suggestion and offer. \n\nThe former two classes are information transfer acts, while the latter two are action discussion acts.")

    sample_id_slider.change(fn=display, inputs=[batch, sample_id_slider], outputs=[context, sample_id_slider, curr_candidate, curr_candidate_annot, usefulness, fluency, emot, emot_adequate, strat_adequate, role_consistency])

demo.launch(share=True, server_port=7860)