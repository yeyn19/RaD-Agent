from copy import deepcopy
import json
import argparse
import os
import re
import traceback
import requests
from termcolor import colored

from calc_reward import calc_reward
from run_webshop.dump_reward import batch_all_buy

def match_value(file_name):
    match = re.search(r'\-?\d+\_', file_name)
    # print(file_name, match)
    result = None
    if match != None:
        result = match.group()[:-1]
    # print(result)
    return result

def count_files(folder_path, start, end):
    file_idx_list = []
    file_count = 0
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".json"):
            file_count += 1
            result = match_value(file_name)
            if result != None:
                result = int(result)
                if result >= start and result < end:
                    file_idx_list.append(int(result))
    # print(f"{folder_path}:", file_count)
    return file_idx_list

def match_file(folder_path, idx):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".json"):
            result = match_value(file_name)
            if result != None:
                if idx == int(result):
                    return file_name

def get_product_attr(file_path):
    try:
        with open(file_path, 'r') as final_f:
            final_json = json.load(final_f)
            train_messages = final_json['answer_generation']['train_messages'][-1][::-1]
            for message in train_messages:
                if "function_call" in message.keys():
                    function_call = message["function_call"]
                    if function_call["name"] == 'buy':
                        args = json.loads(function_call['arguments'])
                        if 'attribute' in args.keys():
                            attribute = str(args["attribute"]).lower()
                        else:
                            attribute = None
                        break
            for message in train_messages:
                if "function_call" in message.keys():
                    function_call = message["function_call"]
                    if function_call["name"] == 'select':
                        args = json.loads(function_call['arguments'])
                        product = str(args["product"]).lower()
                        return product, attribute

                    
    except BaseException as e:
        print(traceback.format_exc(), e)
        return None, None
    return None, None

def get_score(input_dir, mode='first', start = 0, end = 500):
    assert mode in ['first', 'final', 'best']
    file_idx_list = count_files(input_dir, start, end)
    with open(os.path.join(input_dir, "webshop.jsonl"), "r") as fr:
        lines = fr.readlines()
        score_dict = {}
        parse_fail = []
        down_opt = []
        reward_count = 0
        ths = 0.4
        ths = 0.4
        if mode == 'final':
            batch_all_buy(input_dir, "")
        for line in lines:
            reward = float(json.loads(line)['reward'])
            idx = int(json.loads(line)['idx'])
            if idx < start or idx >= end:
                continue
            if idx not in score_dict.keys():
                if mode == 'final':
                    # print(idx)
                    match_file_name = match_file(input_dir, idx)
                    if match_file_name == None:
                        # print("continue")
                        continue
                    match_file_path = os.path.join(input_dir, match_file_name)
                    # open file & find final answer
                    product, attr = get_product_attr(match_file_path)
                    # print(f"product, attr = {(product, attr)}")
                    elo_reward = calc_reward(idx, product, attr, input_dir, output_dir_path='')
                    
                    # if idx == 10:
                    #     print("before reward: ", reward, f", prod: {json.loads(line)['product']}, attr: {json.loads(line)['attribute']}")
                    #     print('after reward: ', elo_reward, f", prod: {product}, attr: {attr}")
                    if elo_reward == None:
                        print(colored(f"WARNING: the reward is not found. idx: {idx}, product: {product}, attribute: {attr}", color='red'))
                        if idx not in parse_fail:
                            parse_fail.append(idx)
                        continue
                    if elo_reward < reward:
                        down_opt.append({'idx': idx, 'before': reward, 'after': elo_reward})
                        if reward - elo_reward > ths:
                            reward_count += 1
                    score_dict[idx] = elo_reward
                elif mode == 'first' or 'best':
                    score_dict[idx] = reward
            else:
                if mode == 'best':
                    if score_dict[idx] < reward:
                        score_dict[idx] = reward
        diff = set(file_idx_list) - set(score_dict.keys())
        miss = [item for item in diff]
        inv_miss = set(score_dict.keys()) - set(file_idx_list)
        if mode == 'final':
            print("score dict: ", score_dict)
            print("parse fail: ", parse_fail)
            good_keys = set(score_dict.keys()) - set(parse_fail)
            good_list = [score_dict[key] for key in good_keys]
            print("good_list average: ", sum(good_list) / len(good_list))
            print("neg optimize: ", len(down_opt), '\n', down_opt)
            print(f" > {ths} :", reward_count)
            # if no these neg:
            fixed_dict = deepcopy(score_dict)
            fix_count = 0
            fix_list = []
            for item in down_opt:
                idx = item['idx']
                if item['before'] - item['after'] > ths:
                    fix_count += 1
                    fixed_dict[idx] = item['before']
                    fix_list.append(item)
            fixed_list = [fixed_dict[key] for key in fixed_dict.keys()]
            print(f"fixed len: {fix_count}, fixed average: {sum(fixed_list) / len(fixed_list)}\n fixed items: {fix_list}")
        return score_dict, file_idx_list, miss, inv_miss

def cross_get_score(output_dir1, output_dir2, choose_best=False):
    score_dict1, _, _, _ = get_score(output_dir1, choose_best)
    score_dict2, _, _, _ = get_score(output_dir2, choose_best)
    same_part = set(score_dict1.keys()) & set(score_dict2.keys())
    score_list1 = [score_dict1[idx] for idx in same_part]
    score_list2 = [score_dict2[idx] for idx in same_part]
    return score_list1, score_list2, same_part

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir1', type=str, default="", required=True, help='output dir1')
    parser.add_argument('--output_dir2', type=str, default="", required=False, help='output dir2')
    parser.add_argument('--start', type=int, default=0, required=False, help='')
    parser.add_argument('--end', type=int, default=500, required=False, help='')
    args = parser.parse_args()
    end = args.end
    start = args.start
    output_dir1 = args.output_dir1
    output_dir2 = args.output_dir2
    mode = "first"

    if output_dir2 == "":
        score_dict, file_idx_list, miss, inv_miss = get_score(output_dir1, mode=mode, start=start, end=end)
        score_list = score_dict.values()
        all_len = end - start
        print(f"all len: {len(file_idx_list)}")
        print(f"success len: {len(score_list)}")
        print(f"miss: {miss}")
        print(f"inv miss: {inv_miss}")
        print(f"all average: {sum(score_list) / len(file_idx_list)}")
        print(f"success average: {sum(score_list) / len(score_list)}")
    else:
        score_dict1, score_dict2, same_part= cross_get_score(output_dir1, output_dir2)
        print(f"all average1: {sum(score_dict1) / len(same_part)}")
        print(f"success average1: {sum(score_dict1) / len(score_dict1)}")
        print(f"all average2: {sum(score_dict2) / len(same_part)}")
        print(f"success average2: {sum(score_dict2) / len(score_dict2)}")
