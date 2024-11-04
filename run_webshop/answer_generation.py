'''
categorytool
'''
import sys
import traceback
sys.path.append("../")

from utils import DATA_DIR

from LLM.fake_llm import chatGPT
from LLM.openai_0613 import chatgpt_0613
from Algorithms.single_chain import single_chain
from Algorithms.reflexion import reflexion_chain
from Algorithms.BFS import BFS_tree_search
from Algorithms.DFS import DFS_tree_search
from Algorithms.UCT_vote_function import UCT_vote_function
from Algorithms.ETS import ETS_tree_search
from Downstream_tasks.base_env import base_env
from utils import standardize, change_name

import re
import os
import time
import json
import jsonlines
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import signal
from termcolor import colored
import requests
from typing import overload
from functools import partial
from pprint import pprint
import pdb
import random
from tqdm import tqdm
import pickle
import queue
import threading
from copy import deepcopy


def api_json_to_openai_json(api_json,standard_tool_name):
    description_max_length=256
    templete =     {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": [],
            "optional": [],
        }
    }
    # rapidapiopenai function
    map_type = {
        "NUMBER": "integer",
        "STRING": "string",
        "BOOLEAN": "boolean"
    }

    pure_api_name = change_name(standardize(api_json["api_name"]))
    templete["name"] = pure_api_name+ f"_for_{standard_tool_name}"
    templete["name"] = templete["name"][-64:] #64

    templete["description"] = f"This is the subfunction for tool \"{standard_tool_name}\", you can use this tool."
    

    if api_json["api_description"].strip() != "":
        tuncated_description = api_json['api_description'].strip().replace(api_json['api_name'],templete['name'])[:description_max_length]
        templete["description"] = templete["description"] + f"The description of this function is: \"{tuncated_description}\""
    if "required_parameters" in api_json.keys() and len(api_json["required_parameters"]) > 0:
        for para in api_json["required_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else: # string
                param_type = "string"
            prompt = {
                "type":param_type,
                "description":para["description"][:description_max_length],
            }

            default_value = para['default']
            if len(str(default_value)) != 0:    
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["required"].append(name)
        for para in api_json["optional_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else: # string
                param_type = "string"

            default_value = para['default']
            if len(str(default_value)) != 0:    
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["optional"].append(name)

    return templete, api_json["category_name"],  pure_api_name

def method_converter(backbone_model,method,env,process_id,single_chain_max_step=24,max_query_count=60):
    if backbone_model == "ChatGPT":
        model = "gpt-3.5-turbo-16k-0613"
    elif backbone_model == "GPT4":
        # model = "gpt-4-32k-0613"
        model = "gpt-4-0613"
    else:
        print(f"Unsupported model: {backbone_model}")
        raise NotImplementedError
    
    llm_forward = chatgpt_0613(model=model)
    if method.startswith("CoT"): #UCTCoT
        passat = int(method.split("@")[-1])
        chain = single_chain(llm=llm_forward, io_func=env,process_id=process_id)
        result = chain.start(
                            pass_at=passat,
                            single_chain_max_step=single_chain_max_step,
                            max_query_count=max_query_count,
                            answer=1)
    elif method.startswith("Reflexion"):
        passat = int(method.split("@")[-1])
        chain = reflexion_chain(llm=llm_forward, io_func=env,process_id=process_id)
        result = chain.start(
                            max_chain_count=passat,
                            single_chain_max_step=single_chain_max_step,
                            max_query_count=max_query_count)
    elif method.startswith("DFS"):
        # print(method)
        pattern = r"DFS(.*)_w(\d+)"
        re_result = re.match(pattern,method)
        assert re_result != None
        subfix = re_result.group(1)
        width = int(re_result.group(2))
        with_filter = True
        if "woFilter" in subfix:
            with_filter = False
        chain = DFS_tree_search(llm=llm_forward, io_func=env,process_id=process_id)
        result = chain.start(
                            single_chain_max_step=single_chain_max_step,
                            tree_beam_size = width,
                            max_query_count = max_query_count,
                            answer=3,
                            with_filter=with_filter)
    elif method.startswith("BFS"):
        pattern = r"BFS(.*)_w(\d+)_e(\d+)"
        re_result = re.match(pattern,method)
        assert re_result != None
        subfix = re_result.group(1)
        width = int(re_result.group(2))
        expansion_ratio = int(re_result.group(3))

        chain = BFS_tree_search(llm=llm_forward, io_func=env,process_id=process_id)
        result = chain.start(
                            single_chain_max_step=single_chain_max_step,
                            tree_beam_size = width,
                            max_query_count = max_query_count,
                            expansion_ratio = expansion_ratio,
                            answer=3,
                            )
    elif method == "UCT_vote":
        chain = UCT_vote_function(llm=llm_forward, io_func=env,process_id=process_id)
        result = chain.start(simulation_count=5,
                            epsilon_new_node=0.3,
                            choice_count=3,
                            vote_candidates=2,
                            vote_count=1,
                            single_chain_max_step=single_chain_max_step,
                            max_query_count = max_query_count)
    elif method.startswith("ETS"):
        pattern = r"ETS(.*)_s(\d+)_f(\d+)_t([\d\.]+)_p([\d\.]+)_c(\d+)_m(\d+)_rn(\d+)_rg(\d+)"
        re_result = re.match(pattern,method)
        if re_result != None:
            subfix = re_result.group(1)
            # print(subfix)
            global_selction_method = "random"
            if "annealing" in subfix:
                global_selction_method = "annealing"
            simulation_count = int(re_result.group(2))
            filter_size = int(re_result.group(3))
            temperature = float(re_result.group(4))
            p_new_node = float(re_result.group(5))
            max_child_count = int(re_result.group(6))
            matching_interval = int(re_result.group(7))
            new_candidate_race_count = int(re_result.group(8))
            global_race_count = int(re_result.group(9))
            chain = ETS_tree_search(llm=llm_forward, io_func=env,process_id=process_id)
            result = chain.start(simulation_count=simulation_count,
                                p_new_node=p_new_node,
                                max_child_count=max_child_count,
                                temperature=temperature,
                                filter_size = filter_size,
                                matching_interval=matching_interval,
                                single_chain_max_step=single_chain_max_step,
                                max_query_count = max_query_count,
                                Elo_args={"k":50,
                                        "new_candidate_race_count": new_candidate_race_count,
                                        "global_race_count":global_race_count,
                                        "global_selction_method": global_selction_method, # random annealing
                                        "temperature":temperature,
                                        }
                                )
        else:
            print(f"method name error: {method} not in {pattern}")
            raise NotImplementedError
    else:
        print("invalid method")
        raise NotImplementedError
    return chain, result

# from webshopping_env import webshopping_env
from webshopping_env_online import webshopping_env

def run(method, backbone_model, query_id, data_dict, query, output_dir_path,tool_des,process_id=0):
    try:
        splits = output_dir_path.split("/")
        os.makedirs("/".join(splits[:-1]),exist_ok=True)
        os.makedirs("/".join(splits),exist_ok=True)
    
        output_file_path = os.path.join(output_dir_path,f"{query_id}_{backbone_model}_{method}.json")
        if os.path.exists(output_file_path):
            return
    
        env = webshopping_env(query_id, output_dir_path,process_id=process_id)
    
        if process_id == 0:
            print(colored(f"[process({process_id})]now playing {env.input_description}, with {len(env.functions)} APIs", "green"))
        max_query_count = 100
        chain,result = method_converter(
            backbone_model=backbone_model,
            method=method,
            env=env,
            process_id=process_id,
            single_chain_max_step=12,
            max_query_count=max_query_count)
        if output_dir_path is not None:
            if chain.json_list != None and False:
                save_interval = 30
                expected_len = max_query_count // save_interval
                
                if chain.query_count < max_query_count:
                    chain.json_list.append(chain.to_json(answer=True,process=True))
                assert len(chain.json_list) <= expected_len
                cont = chain.json_list[-1]
                while len(chain.json_list) < expected_len:
                    chain.json_list.append(deepcopy(cont))
                for i in range(len(chain.json_list)):
                    data = chain.json_list[i]
                    with open(os.path.join(output_dir_path,f"{query_id}_{backbone_model}_{method}_{(i+1)*save_interval:04d}.json"),"w") as writer:
                        data["answer_generation"]["query"] = query
                        json.dump(data, writer, indent=2)
                        success = data["answer_generation"]["valid_data"] and "give_answer" in data["answer_generation"]["final_answer"]
                        print(colored(f"[process({process_id})]valid={success}", "green"))
            else:
                with open(output_file_path,"w") as writer:
                    data = chain.to_json(answer=True,process=True)
                    data["answer_generation"]["query"] = query
                    json.dump(data, writer, indent=2)
    
                    success = data["answer_generation"]["valid_data"] and "give_answer" in data["answer_generation"]["final_answer"]
                    print(colored(f"[process({process_id})]valid={success}", "green"))
    except BaseException as e:
        raise e

    return result


def contain(candidate_list,white_list):
    '''
    candidate_listwhite_list
    '''
    output = []
    for cand in candidate_list:
        if cand not in white_list.keys():
            return False
        output.append(white_list[cand])
    return output


def main(query_dir, answer_dir, method,backbone_model):
    
    task_list = []
    idx_list = [i for i in range(500)]

    for k,idx in tqdm(enumerate(idx_list)):
        task_list.append((method, backbone_model, idx, {}, "", answer_dir, ""))
    return task_list

def get_white_list():
    white_list_dir = os.path.join(DATA_DIR,"jsons_filtered_pipeline_subscribed_white_list")
    white_list = {}
    for cate in tqdm(os.listdir(white_list_dir)):
        for file in os.listdir(os.path.join(white_list_dir,cate)):
            assert file.endswith(".json")
            standard_tool_name = file.split(".")[0]
            with open(os.path.join(white_list_dir,cate,file)) as reader:
                js_data = json.load(reader)
            origin_tool_name = js_data["tool_name"]
            # white_list.append(standard)
            white_list[origin_tool_name] = {"description": js_data["tool_description"], "standard_tool_name": standard_tool_name}
    
    return white_list


class Consumer(threading.Thread):

    def __init__(self, process_id,starting_time, stop_event:threading.Event):
        super().__init__()
        self.process_id = process_id
        self.starting_time=starting_time
        self.stop_event = stop_event

    def run(self):
        global q
        while not q.empty():
            task=q.get()       #\
            print(f"process[{self.process_id}] get task, now task_queue len={q.qsize()}, time_usage={(time.time() - self.starting_time)/60:.2f}min")

            try:
                run(*task,process_id=self.process_id)
                # exit(1)
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    self.stop_event.set()
                    exit(6)
                else:
                    traceback.print_exc()
                    print(e)
                    raise e
        print(f"process[{self.process_id}] finish, time_usage={(time.time() - self.starting_time)/60:.2f}min")

if __name__ == "__main__":
    try:
        answer_file_name = "answer_G3_singleanswer"
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_query_file', type=str, default=os.path.join(DATA_DIR,"queryG3.json"), required=False, help='input path')
        parser.add_argument('--output_answer_file', type=str, default=os.path.join(DATA_DIR,answer_file_name), 
        required=False, help='output path')
    
        parser.add_argument('--process_num', type=int, default=60, required=False, help='process number')
        parser.add_argument('--debug', type=int, default=0, required=False, help='1 for debugging')
        parser.add_argument('--backbone_model', type=str, default="ChatGPT", required=False, help='backbone_model')
    
        parser.add_argument('--method', type=str, default="DFS_woFilter_w2", required=False, help='method for answer generation: CoT@n,Reflexion@n,BFS_wn_en, DFS_woFilter_wn,UCT_vote, ETS_sn_fn_tn_pn_cn_mn_rnn_rgn') 


        args = parser.parse_args()
    
        query_dir = args.input_query_file
        answer_dir = args.output_answer_file
        method = args.method
        process_num = args.process_num
        debug = args.debug
        backbone_model = args.backbone_model
    
    
        task_list = main(query_dir, answer_dir, method,backbone_model)

        random.seed(42)
        random.shuffle(task_list)
    
        valid_query_id_file_name = "20000"
    
    
        total_task_len = len(task_list)

        print(f"total tasks: {len(task_list)}")
    
        new_task_list = []
        valid_query_id = []
        for task in tqdm(task_list):
            out_dir_path = task[-2]
            query_id = task[2]
            
    
            valid_query_id.append(query_id)
            output_file_path = os.path.join(out_dir_path,f"{query_id}_{backbone_model}_{method}.json")
            if not os.path.exists(output_file_path):
                new_task_list.append(task)
        task_list = new_task_list
    
        undo_task_len = len(task_list)

        if undo_task_len == total_task_len:
            pass


        print(f"undo tasks: {len(task_list)}")
    

        stop_event = threading.Event()

        q=queue.Queue(10000000)  #
        starting_time = time.time()
        for task in task_list: #
            q.put(task)
    
        # 
        thread_list = []
        for i in range(process_num):
            p = Consumer(process_id=i,starting_time=starting_time, stop_event=stop_event)
            p.start()
            thread_list.append(p)

        while thread_list:
            for t in thread_list:
                if not t.is_alive():
                    thread_list.remove(t)

        exit(1)

    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            print(traceback.format_exc(), str(e))
            exit(6)
        elif isinstance(e, SystemExit):
            exit(e.code)
        else:
            print(traceback.format_exc(), str(e))