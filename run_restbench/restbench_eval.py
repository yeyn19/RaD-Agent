from copy import deepcopy
import json
import os
import re

def find_matching_string(pattern, string):
    regex_pattern = re.sub(r'.*\{.*?\}.*', r'.*', pattern)

    regex = re.compile(regex_pattern)

    return regex.search(string) is not None

data_dir = "data/tmdb/output/"

result_name = "restbench.jsonl"

ref_result_path = 'RestGPT/datasets/tmdb.json'

output_path = "restbench_result.json"

result_dict = {}
line_count = 0
with open(os.path.join(ref_result_path), "r", encoding='utf-8') as ref_fr:
    ref_data = json.load(ref_fr)
    with open(os.path.join(data_dir, result_name), "r", encoding='utf-8') as fr:
        str_line = fr.readline()
        while str_line:
            str_data = json.loads(str_line)
            # print(str_data)
            idx = str_data['idx']
            gold_res = ref_data[idx]['solution']
            
            # result_dict['idx'] = idx
            if idx not in result_dict:
                result_dict[idx] = {}

            result_dict[idx]['gold_res'] = gold_res
            
            if 'flags' not in result_dict[idx]:
                flags = [False] * len(gold_res)
                result_dict[idx]['flags'] = flags
            
            for gold_idx, gold_item in enumerate(gold_res):
                for his_item in str_data['history']:
                    traj_action = his_item['action']
                    if traj_action['action'] == 'request':
                            
                        split_gold_item = gold_item.split(" ")
                        gold_method, gold_url = split_gold_item[0], split_gold_item[1]
                        
                        method_flag = False
                        if traj_action['method'] == gold_method:
                            method_flag = True
                        
                        url_flag = False
                        if 'payload' in traj_action and 'url' in traj_action['payload']:
                            if find_matching_string(gold_url, traj_action['payload']['url']):
                                url_flag = True

                        if method_flag and url_flag:
                            result_dict[idx]['flags'][gold_idx] = True
                            break
            # result_list.append(deepcopy(result_dict))
            line_count += 1
            str_line = fr.readline()
print("line_count: ", line_count)

wrong_list = []
count = 0
for idx, result in result_dict.items():
    final_flag = True
    for flag_item in result['flags']:
        if not flag_item:
            final_flag = False
            wrong_list.append(idx)
            break
    result_dict[idx]['correct'] = final_flag
    if final_flag:
        count += 1

set_a = set(result_dict.keys())
set_full = set([i for i in range(100)])
print(set_a)
set_miss = set_full - set_a
print(set_miss)

print(f"correct: {count}/{len(result_dict)}, acc: {count/len(result_dict)}")
print(f"wrong list: {wrong_list}")
with open(os.path.join(output_path), "w", encoding='utf-8') as fw:
    json.dump(result_dict, fw)
    