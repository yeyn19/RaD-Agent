import json
from termcolor import colored
import os
lines = []

def calc_reward(idx, prod, attr, in_dir_path: str, output_dir_path: str, in_file_name="shop_reward.jsonl"):
    '''
    according to the cand actions, calculate the real reward.
    '''
    # print(f"calculate idx: {idx}'s reward")
    # env = webshopping_env(goal_idx=idx, output_dir_path=output_dir_path)
    # idx = check_right(idx, env.input_description, chain)
    # if idx == None:
    #     print(f"failed to calculate: query not found")
    #     return None
    # print(f'continue to calculate {idx}\'s score')
    # print(f"{env.input_description}")
    # env.reward = 0
    # action_name = None
    # for node in chain:
    #     # print(node)
    #     if env.check_success == 1:
    #         break
    #     if node['type'] == 'Action':
    #         action_name = node['description']
    #     elif node['type'] == 'Action Input':
    #         action_input = node['description']
    #         print(f"action name: {action_name}, action input: {action_input}")
    #         env.step(action_name=action_name, action_input=action_input)
    # return env.reward
    global lines
    if len(lines) == 0:
        with open(os.path.join(in_dir_path, in_file_name), "r") as fr:
            lines = fr.readlines()
    # match reward
    reward = None
    if attr == None:
        attr = ''
    for line in lines:
        line_json = json.loads(line)
        if int(line_json['idx']) == int(idx):
            # print(f"check line json idx: {idx}")
            if line_json['product'] == prod and line_json['attribute'] == attr:
                reward = line_json['reward']
    return reward
