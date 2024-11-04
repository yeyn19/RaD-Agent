import os
import json
import argparse
from run_webshop.env_utils import get_buy_info_from_json

def get_buy_info(line):
    '''
    convert single line json into buy info
    '''
    line_json = json.loads(line)
    return get_buy_info_from_json(line_json)

def batch_all_buy(data_dir, input_dir, file_name='webshop.jsonl'):
    '''
    
    '''
    dir_path=os.path.join(data_dir, input_dir)
    write_path = os.path.join(dir_path, 'shop_reward.jsonl')
    if os.path.exists(write_path):
        os.remove(write_path)
    with open(os.path.join(dir_path, file_name), "r") as fr:
        lines = fr.readlines()
        count = 0
        for line in lines:
            # print(count)
            count = count + 1
            result = get_buy_info(line=line)
            if result['buy']:
                with open(os.path.join(dir_path, 'shop_reward.jsonl'), "a") as fw:
                    json.dump(result, fw)
                    fw.write('\n')

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', required=True, help='')
    parser.add_argument('--input_dir', type=str, default='', required=True, help='')
    args = parser.parse_args()
    input_dir = args.input_dir
    data_dir = args.data_dir
    batch_all_buy(data_dir, input_dir, 'webshop.jsonl')
