'''
生成多category多tool的数据
'''
import sys
sys.path.append("./LLM-AlphaGo/")

from ets_utils import DATA_DIR, CODE_DIR, standardize, change_name

# from LLM.fake_llm import chatGPT
from LLM.openai_0613 import chatgpt_0613
from Algorithms.single_chain import single_chain
from Algorithms.reflexion import reflexion_chain
from Algorithms.BFS import BFS_tree_search
from Algorithms.DFS import DFS_tree_search
from Algorithms.UCT_vote_function import UCT_vote_function
from Algorithms.ETS2 import ETS_tree_search
from MCTS import do_24
from Downstream_tasks.base_env import base_env

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


def add_space(string):
    string = [" "+char for char in string]
    string = "".join(string)[1:]
    string = string.replace("  "," ")
    return string

def int_list_to_string(int_list):
    assert type(int_list) == list

    output = str(int_list)
    char_list = [" "+char for char in output]
    char_list = "".join(char_list)
    char_list = char_list.replace("  "," ").strip()
    return char_list


class wrap_play_24(base_env):
    def __init__(self, query, process_id=0):
        '''
        24点任务
        '''
        super(wrap_play_24).__init__()

        self.int_list = query.split(" ")
        self.now_datas = [int(cont) for cont in self.int_list]

        self.process_id = process_id

        self.task_description = '''Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number 24. Each step, you are only allowed to choose two of the left numbers to obtain a new number. For example, 7 * 9 - 3 * 1 3 = 2 4.
Remember:
1. all of the number must be used, and must be used ONCE. So Only when left numbers is exact 24, you will win. So you don't succeed when left number = [2 4, 5]. You succeed when left number = [2 4]. 
2. all the try takes exactly 3 steps, and the task ends when the count of left numbers if 1, and check whether the only left number equals 24.
3. When there are ONLY two numbers left, ALWAYS pre-compute and list all the operations' combine results( + - * / ), and find if there is a way to combine to 24 before you make the function call. 
3.1. If There is a way, use function "play_24" to combine it.
3.2. If not, use function "give_up" to restart.
4. The status change ONLY when you call function "play_24". If you only give thoughts, nothing happens.
5. "play_24" inputs only one step, if you want to combine many numbers, split it into multiple calls. 
Here is an example:
************************************************************
Example input: [ 3 , 1 3 , 9 , 7 ]

Thought: Now left numbers are [ 3 , 1 3 , 9 , 7 ]. There are so many possible combinations, for example: 
3 + 1 3, left [ 1 6 , 9 , 7 ]
9 * 7, left [ 6 3 , 3 , 1 3 ] 
1 3 - 9, left [ 4 , 3 , 7 ]
...I wll try one of them this time. 

function_call: {
    "name": "play_24",
    "arguments": {
        "input": "3 * 1 3 = 3 9"
    }
}
function_output: 3 * 1 3 = 3 9, left numbers: [ 9 , 7 , 3 9 ]

USER: This is not the first time you try this task, all previous trails failed.
Before you generate my thought for this state, I will first show you your previous actions for this state, and then you must generate actions that is different from all of them. Here are all the previous candidates' function calls from this state:
[
    function_call: {
        "name": "play_24",
        "arguments": {
            "input": "9 + 7 = 1 6"
        }
        "function_output": "9 + 7 = 1 6, left numbers: [ 1 6 , 3 9 ]"
    }
    function_call: {
        "name": "play_24",
        "arguments": {
            "input": "3 9 * 7 = 2 7 3"
        }
        "function_output": "3 9 * 7 = 2 7 3, left numbers: [ 9 , 2 7 3 ]"
    }
]
Remember:
1.You are now in the intermediate state of a trail. You will first repeat your now states.
2.Then you will repeat all the previous candidates' function call and results from this state, one candidates one line.
3.Finally, You will make a function call that is different from all the previous candidates'.
Again, Now your state is: 
3 * 1 3 = 3 9, left numbers: [ 9 , 7 , 3 9 ]

Thought: I repeat user's prompt first: Now left numbers are [ 9 , 7 , 3 9 ].
There are 3 numbers left. All the previous candidates in user's instruction do the following things:
9 + 7 = 1 6, left [ 1 6 , 3 9 ]
3 9 * 7 = 2 7 3, left numbers: [ 9 , 2 7 3 ]
So as for this time, I will make a different function call, I should use *. I shouldn't use 7*9 because previous candidates tried it. Let's try 7*9.

function_call: {
    "name": "play_24",
    "arguments": {
        "input": "7 * 9 = 6 3"
    }
}

function_output: 7 * 9 = 6 3, left numbers: [ 3 9 , 6 3 ]

Thought: Now left numbers are [ 3 9 , 6 3 ]. There are only 2 numbers now. I will pre-compute and list all the possible actions when there are only 2 numbers left: 
- 6 3 + 3 9 = 1 0 2, not 2 4 
- 3 9 + 6 3 = 1 0 2, not 2 4
- 6 3 - 3 9 = 2 4,  success!
- 3 9 - 6 3 = - 2 4, not 2 4
- 6 3 * 3 9 = 2 4 5 7, not 2 4 
- 3 9 * 6 3 = 2 4 5 7, not 2 4 
- 6 3 / 3 9 is not a integer, not 2 4
- 3 9 / 6 3 is not a integer, not 2 4
I find a Solution!

function_call: {
    "name": "play_24",
    "arguments": {
        "input": "6 3 - 3 9 = 2 4"
    }
}

function_output: 6 3 - 3 9 = 2 4, left numbers: [ 2 4 ], you win!
************************************************************
'''

        self.input_description = f'''Now, your input is {int_list_to_string(self.now_datas)}'''

        self.status = 0

        self.functions = [
            {
                "name": "play_24",
                "description": "make your current conbine with the format \"x operation y = z\" like \"1 + 2 = 3\", then I will tell you whether you win. This is the ONLY way to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "describe what number you want to conbine, and how to conbine.",
                        },

                    },
                    "required": ["input"],
                },
            },
            {
                "name": "give_up",
                "description": "If you think you can't handle the task from this status, call this function to restart. You can call this function ONLY when there are two numbers left",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": [],
                }
            }
        ]

    def check_success(self):
        return self.status == 1

    def to_json(self):
        return {}

    def restart(self):
        pass

    def get_score(self):
        return 0.0

    def have_hope(self) -> bool:
        """目前的self.now_datas是否已经无解了"""
        return do_24(self.now_datas, [])

    def step(self,action_name, action_input):
        '''
        action_name 用的是没有normalize的版本，先找找有没有
        需要返回一个observation string以及状态码：
        0代表正常返回
        1代表没有对应api名字
        2代表输入有错误
        3代表生成结束，出现final answer
        4代表模型自己决定剪枝
        '''
        if action_name == "give_up":
            return "{\"response\":\"You choose to restart the task\"}", 4
        
        try:
            if type(action_input) == str:
                js_data = json.loads(action_input)
                assert "input" in js_data.keys()
                action_input = js_data["input"]
            else:
                action_input = action_input["input"]
        except Exception as e:
            pass
        action_input = action_input.replace(" ","")
        former_data_str = f"{int_list_to_string(self.now_datas)}"
        re_pattern = r"(\(?(\-?\d+\.?\d*)\)?[\+\-\*\/]\(?(\-?\d+\.?\d*)\)?)=(\-?\d+)"
        # re_pattern = r"((\-?\(?[\d\.]+\)?)([\+\-\*\/ ]+?)(\-?\(?[\d\.]+\)?))[^\+\-\*\/=]*=\D*(\-?[\d\.]+).*"
        match_result = re.match(re_pattern,action_input)
        if match_result != None:
            d1 = eval(match_result.group(2))
            d2 = eval(match_result.group(3))
            '''
            观察数字是否存在
            '''
            if d1 not in self.now_datas:
                return f"{d1} not in left numbers {former_data_str}", 2
            if d2 not in self.now_datas:
                return f"{d2} not in left numbers {former_data_str}", 2
            if d1 == d2 and self.now_datas:
                counter = list( filter(lambda x:x==d1,self.now_datas))
                if len(counter) < 2:
                    return f"there are only one \"{d1}\" number in left numbers {former_data_str}", 2

            try:
                result = eval(match_result.group(1))
                # if type(result) == float and not result.is_integer():
                #     return f"you combine numbers to {result}, which is not a integer, try other command. left numbers {former_data_str}", 0
                '''
                数字存在，筛出数字
                '''
                # result = int(result)
                ls1 = self.now_datas.index(d1)
                self.now_datas = self.now_datas[:ls1] + self.now_datas[ls1+1:]
                ls2 = self.now_datas.index(d2)
                self.now_datas = self.now_datas[:ls2] + self.now_datas[ls2+1:] 
                self.now_datas.append(result)  
                formula = f"{match_result.group(1)}={result}"
                obs = f"{add_space(formula)}, left numbers: {int_list_to_string(self.now_datas)}"

                
                if len(self.now_datas) == 1:
                    if self.now_datas[0] == 24:
                        self.status = 1
                        return f"{obs}, you win", 3
                    else:
                        self.status = 0
                        return f"{obs}, you lose", 4
                else:
                    return obs, 0
            except Exception as e:
                return str(e), 2
        else:
            return '''format error, please use format \"x operations y = result\"''', 2


def method_converter(backbone_model,method,env,process_id,single_chain_max_step=24,max_query_count=60):
    
    llm_forward = chatgpt_0613(model=backbone_model)
    if method.startswith("CoT"): #单次模拟的UCT就是CoT
        pattern = r"CoT@(\d+)([^\d]*)"
        re_result = re.match(pattern,method)
        assert re_result != None
        passat = int(re_result.group(1))
        subfix = re_result.group(2)
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
                            answer=1,
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
                            answer=1,
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
            print(subfix)
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
                                        },
                                subfix=subfix,
                                )
        else:
            print(f"method name error: {method} not in {pattern}")
            raise NotImplementedError
    else:
        print("invalid method")
        raise NotImplementedError
    return chain, result


def run(method, backbone_model, query_id, query, output_dir_path, process_id=0):
    splits = output_dir_path.split("/")
    os.makedirs("/".join(splits[:-1]),exist_ok=True)
    os.makedirs("/".join(splits),exist_ok=True)

    output_file_path = os.path.join(output_dir_path,f"{query_id}_{backbone_model}_{method}.json")
    if os.path.exists(output_file_path):
        return

    env = wrap_play_24(query, process_id=process_id)

    if process_id == 0:
        print(colored(f"[process({process_id})]now playing {query}, with {len(env.functions)} APIs", "green"))
    max_query_count = 300
    chain,result = method_converter(
        backbone_model=backbone_model,
        method=method,
        env=env,
        process_id=process_id,
        single_chain_max_step=18,
        max_query_count=max_query_count)
    if output_dir_path is not None:
        with open(output_file_path,"w") as writer:
            data = chain.to_json(answer=True,process=True)
            data["answer_generation"]["query"] = query
            json.dump(data, writer, indent=2)

            success = data["win"]
            print(colored(f"[process({process_id})]valid={success}", "green"))

    return result


def main(query_dir, answer_dir, method,backbone_model):
    # 每个工具只先跑一条query
    task_list = []
    with open(query_dir,"r") as reader:
        for k, line in enumerate(reader.readlines()[1:]):
            if k not in range(900,1000):
                continue
            question = line.split(",")[1]
            task_list.append((method, backbone_model, k, question, answer_dir))

    return task_list


class Consumer(threading.Thread):

    def __init__(self, process_id,starting_time):
        super().__init__()
        self.process_id = process_id
        self.starting_time=starting_time

    def run(self):
        global q
        while not q.empty():
            task=q.get()       #默认阻塞\
            print(f"process[{self.process_id}] get task, now task_queue len={q.qsize()}, time_usage={(time.time() - self.starting_time)/60:.2f}min")
            run(*task,process_id=self.process_id)
        print(f"process[{self.process_id}] finish, time_usage={(time.time() - self.starting_time)/60:.2f}min")

if __name__ == "__main__":
    # re_pattern = r"(\(?(\-?\d+\.?\d*)\)?[\+\-\*\/]\(?(\-?\d+\.?\d*)\)?)=(\-?\d+)"
    # result = re.match(re_pattern, "(-16.6)-(-1.7)=17")
    # print(result.groups())
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_query_file', type=str, default=os.path.join(CODE_DIR,"Downstream_tasks","24.csv"), required=False, help='input path')
    parser.add_argument('--output_answer_file', type=str, default=os.path.join(DATA_DIR,"play24_0806_4mini"), 
    required=False, help='output path')
    print(DATA_DIR)

    # parser.add_argument('--input_query_file', type=str, default=os.path.join(DATA_DIR,"queryG3.json"), required=False, help='input path')
    # parser.add_argument('--output_answer_file', type=str, default=os.path.join(DATA_DIR,"multicate_multitool_multianswer_new_format"), 
    # required=False, help='output path')

    parser.add_argument('--process_num', type=int, default=1, required=False, help='process number')
    parser.add_argument('--debug', type=int, default=0, required=False, help='1 for debugging')
    parser.add_argument('--backbone_model', type=str, default="gpt-4o-mini", required=False, help='backbone_model')

    parser.add_argument('--method', type=str, default="ETS_all-100_annealing_k50_sqrt_s100_f1_t173.72_p0.9_c15_m3_rn1_rg3", required=False, help='method for answer generation: CoT@n, Reflexion@n, BFS_wn_en, DFS_woFilter_wn, UCT_vote, ETS_annealing_sqrt_s10_f1_t173.72_p0.5_c8_m3_rn3_rg4') 
    # DFS_woFilter_w7
    # CoT@100
    # elo 匹配算法调整 ETS_randomselect_k50_sqrt_s100_f1_t173.72_p0.9_c15_m3_rn1_rg3
    # elo全-100 ETS_all100_annealing_k50_sqrt_s100_f1_t173.72_p0.9_c15_m3_rn1_rg3
    # elo初值调整 ETS_randomelo_annealing_k50_sqrt_s100_f1_t173.72_p0.9_c15_m3_rn1_rg3
    # ETS_annealing_k50_sqrt_s100_f1_t173.72_p0.9_c15_m3_rn1_rg3
    # ETS_annealing_k50_sqrt_s100_f1_t173.72_p0.9_c24_m1_rn0_rg0

    args = parser.parse_args()

    query_dir = args.input_query_file
    answer_dir = args.output_answer_file
    method = args.method
    process_num = args.process_num
    debug = args.debug
    backbone_model = args.backbone_model


    task_list = main(query_dir, answer_dir, method,backbone_model)

    
    # random.seed(42)
    # random.shuffle(task_list)


    print(f"total tasks: {len(task_list)}")

    new_task_list = []
    valid_query_id = []
    for task in tqdm(task_list):
        out_dir_path = task[-1]
        query_id = task[2]
        valid_query_id.append(query_id)
        # import pdb; pdb.set_trace()
        output_file_path = os.path.join(out_dir_path,f"{query_id}_{backbone_model}_{method}.json")
        # print(output_file_path)
        if not os.path.exists(output_file_path):
            new_task_list.append(task)
    task_list = new_task_list

    print(f"undo tasks: {len(task_list)}")


    q=queue.Queue(10000000)  #创建一个先进先出的队列
    starting_time = time.time()
    for task in task_list: #共享的任务列表，所有线程一起消费
        q.put(task)

    for i in range(process_num):
        p = Consumer(process_id=i,starting_time=starting_time)
        p.start()

