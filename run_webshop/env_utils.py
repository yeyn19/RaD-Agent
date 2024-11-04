
import os
import re
import json

def get_buy_info_from_json(line_json):
    idx = line_json['idx']
    reward = line_json['reward']
    history = line_json['history'][::-1]
    black_list = ['back to search', 'search', 'next >', '< prev', 'description', 'features', 'reviews', 'buy now']
    buy = False
    attribute = ''
    product = ''
    for his in history:
        action = str(his['action'])
        if action.startswith('click'):
            click_name = action[6:-1]
            if click_name == 'buy now':
                buy = True
            elif click_name not in black_list:
                # decide is clicking product or attribute
                clickables = his['available_actions']['clickables']
                if 'buy now' in clickables:
                    # is clicking attribute.
                    attribute = click_name
                else:
                    # is clicking product
                    product = click_name
                    break
    return {
        'idx': idx,
        "buy": buy,
        "product": product,
        'attribute': attribute,
        'reward': reward
    }



def find_product(idx: int, chain: list):
    # find product
    chain = chain[::-1]
    attr = ''
    prod = ''
    for node in chain:
        if node['type'] == 'Action Input':
            des = node['description']
            des_json = json.loads(des)
            if 'attribute' in des_json.keys():
                attr = str(des_json['attribute']).lower()
            elif 'product' in des_json.keys():
                prod = str(des_json['product']).lower()
                break
    # print(colored(f"idx: {idx}, prod: {prod}, attr: {attr}", color='magenta'))
    return idx, prod, attr



def parse_json(test, key="overall_preference"):
    pattern = f'"{key}":\s*(-?\d+)'
    match = re.search(pattern, test)
    dict = {}
    dict[key] = int(match.group(1))
        # print(f"Key: {key}, Value: {value}")
    return dict


def match_file(file_name):
    result = None
    match = re.match(r'\-?\d+\_', file_name)
    if match:
        result = match.group()[:-1]
    return result

def count_files(folder_path, traverse_func):
    file_count = 0
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".json"):
            file_count += 1
            file_idx = match_file(file_name)
            if file_idx:
                print(f'processing {file_idx}')
                traverse_func(file_idx, file_path)
                # filter_candidate(idx=file_idx, file_path=file_path, output_file_dir=folder_path, output_file_name=output_file_name)
    print(f"{folder_path}:", file_count)
    return file_count
