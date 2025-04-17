#THIS SCRIPT GATHERS THE RESPONSES FROM ALL MODELS TO A SAME CONTEXT IN A CSV FILE.

import pandas as pd


#GPT2
references_path_gpt2 = "path/to/results"
references_path_gpt2_large = "path/to/results"

references_path_gpt2_cd1 = "path/to/results"
references_path_gpt2_cd2 = "path/to/results"

references_path_gpt2_large_cd1 = "path/to/results"
references_path_gpt2_large_cd2 = "path/to/results"


#DIALOGPT
references_path_dialoGPT = "path/to/results"
references_path_dialoGPT_large = "path/to/results"

references_path_dialoGPT_cd1  = "path/to/results"
references_path_dialoGPT_cd2 = "path/to/results"

references_path_dialoGPT_large_cd1  = "path/to/results"
references_path_dialoGPT_large_cd2  = "path/to/results"


#BART
references_path_bart = "path/to/results"
references_path_bart_large = "path/to/results"

references_path_bart_cd1 = "path/to/results"
references_path_bart_cd2 = "path/to/results"

references_path_bart_large_cd1 = "path/to/results"
references_path_bart_large_cd2  = "path/to/results"


#BELUGA
references_path_beluga = "path/to/results"    

references_path_beluga_fr_cd1 = "path/to/results"   
references_path_beluga_fr_cd2 = "path/to/results"    

references_path_beluga_pb_cd1 = "path/to/results"    
references_path_beluga_pb_cd2 = "path/to/results"    


dic_models={'GPT2 SMALL NO-CD': references_path_gpt2, 'GPT2 MEDIUM NO-CD': references_path_gpt2_large, 'GPT2 SMALL CD1': references_path_gpt2_cd1, 'GPT2 MEDIUM CD1': references_path_gpt2_large_cd1, 'GPT2 SMALL CD2': references_path_gpt2_cd2, 'GPT2 MEDIUM CD2': references_path_gpt2_large_cd2,
            'DIALOGPT SMALL NO-CD': references_path_dialoGPT, 'DIALOGPT MEDIUM NO-CD': references_path_dialoGPT_large, 'DIALOGPT SMALL CD1': references_path_dialoGPT_cd1, 'DIALOGPT MEDIUM CD1': references_path_dialoGPT_large_cd1, 'DIALOGPT SMALL CD2': references_path_dialoGPT_cd2, 'DIALOGPT MEDIUM CD2': references_path_dialoGPT_large_cd2,
            'BART BASE NO-CD': references_path_bart, 'BART LARGE NO-CD': references_path_bart_large, 'BART BASE CD1': references_path_bart_cd1, 'BART LARGE CD1': references_path_bart_large_cd1, 'BART BASE CD2': references_path_bart_cd2, 'BART LARGE CD2': references_path_bart_large_cd2,
            'BELUGA NO-CD': references_path_beluga, 'BELUGA FR CD1': references_path_beluga_fr_cd1, 'BELUGA FR CD2': references_path_beluga_fr_cd2, 'BELUGA PB CD1': references_path_beluga_pb_cd1, 'BART BASE CD2': references_path_bart_cd2, 'BELUGA PB CD2': references_path_beluga_pb_cd2}


ref_res = pd.read_csv(references_path_bart, encoding='UTF-8')
context = ref_res['input'].tolist()
references = ref_res['actual_responses'].tolist()


final_dic = {'context': context, 'reference': references}
for model in dic_models:
    final_dic[model] = []

for k, v in dic_models.items():
    data = pd.read_csv(v, encoding='UTF-8')
    cols = list(data.columns.values)
    colname = 'prediction' if 'prediction' in cols else "generated_responses"
    #print(k, cols, colname)
    final_dic[k] = data[colname].tolist()


final_df = pd.DataFrame(final_dic)
final_df.to_csv('full_responses_english.csv', index=False, encoding='UTF-8')