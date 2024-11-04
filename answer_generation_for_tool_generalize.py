'''
生成多category多tool的数据
'''
import sys
sys.path.append("./LLM-AlphaGo/")

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
    # 参数类型映射，不然直接用rapidapi的参数传到openai function会无法识别报错
    map_type = {
        "NUMBER": "integer",
        "STRING": "string",
        "BOOLEAN": "boolean"
    }

    pure_api_name = change_name(standardize(api_json["api_name"]))
    templete["name"] = pure_api_name+ f"_for_{standard_tool_name}"
    templete["name"] = templete["name"][-64:] #最后64个字母

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
            else: # 其他参数类型都默认设为string
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
            else: # 其他参数类型都默认设为string
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

class wrap_rapidapi_multicate_multitool(base_env):
    def __init__(self, data_dict, query,tool_descriptions = None,process_id=0):
        '''
        将rapid api封装到LLM-alphaGo的接口，顺便需要将rapid-api描述转换成openai Function传参定义
        '''
        super(wrap_rapidapi_multicate_multitool).__init__()

        self.process_id = process_id

        self.tool_names = []
        self.cate_names = []

        self.input_description = query


        self.functions = []

        self.api_name_reflect = {}

        for k,api_json in enumerate(data_dict["api_list"]):
            standard_tool_name = tool_descriptions[k][0]

           
            openai_function_json,cate_name, pure_api_name = api_json_to_openai_json(api_json,standard_tool_name)
            self.functions.append(openai_function_json)

            self.api_name_reflect[openai_function_json["name"]] = pure_api_name
            self.tool_names.append(standard_tool_name)
            self.cate_names.append(cate_name)

        finish_func = {
            "name": "Finish",
            "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {
                        "type": "string",
                        "enum": ["give_answer","give_up_and_restart"],
                    },
                    "final_answer": {
                        "type": "string",
                        "description": "The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"",
                    }
                },
                "required": ["return_type"],
            }
        }

        self.functions.append(finish_func)
        # self.functions.append(father_tool_func)
        self.CALL_MAX_TIME = 3
        self.task_description = f'''You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:\n'''
        
        unduplicated_reflection = {}
        for standardize_tool_name, tool_des in tool_descriptions:
            unduplicated_reflection[standardize_tool_name] = tool_des

        for k,(standardize_tool_name, tool_des) in enumerate(unduplicated_reflection.items()):
            striped = tool_des[:512].replace('\n','').strip()
            if striped == "":
                striped = "None"
            self.task_description += f"{k+1}.{standardize_tool_name}: {striped}\n"

        self.success = 0

    def check_success(self):
        return self.success

    def to_json(self):
        return {}

    def restart(self):
        pass

    def get_score(self):
        return 0.0


    def step(self,**args):
        '''
        添加一个工作流监控，记录错误类型
        '''
        obs, code = self._step(**args)
        if len(obs) > 1024:
            obs = obs[:1024] + "..."
        return obs, code

    def _step(self, action_name="", action_input=""):
        '''
        action_name 用的是没有normalize的版本，先找找有没有
        需要返回一个observation string以及状态码：
        0代表正常返回
        1代表没有对应api名字
        2代表输入有错误
        3代表生成结束，出现final answer
        4代表模型自己决定剪枝
        '''
        if action_name == "Finish":
            # 加个try，有时候finish时的action input转json会报错，比如最后少了个}
            try:
                json_data = json.loads(action_input,strict=False)
            except: # 手动提取信息
                json_data = {}
                if '"return_type": "' in action_input:
                    if '"return_type": "give_answer"' in action_input:
                        return_type = "give_answer"
                    elif '"return_type": "give_up_and_restart"' in action_input:
                        return_type = "give_up_and_restart"
                    else:
                        return_type = action_input[action_input.find('"return_type": "')+len('"return_type": "'):action_input.find('",')]
                    json_data["return_type"] = return_type
                if '"final_answer": "' in action_input:
                    final_answer = action_input[action_input.find('"final_answer": "')+len('"final_answer": "'):]
                    json_data["final_answer"] = final_answer
            # print(json_data)
            if "return_type" not in json_data.keys():
                return "{error:\"must have \"return_type\"\"}", 2
            if json_data["return_type"] == "give_up_and_restart":
                return "{\"response\":\"chose to give up and restart\"}",4
            elif json_data["return_type"] == "give_answer":
                if "final_answer" not in json_data.keys():
                    return "{error:\"must have \"final_answer\"\"}", 2
                
                self.success = 1 #成功返回 final_answer
                return "{\"response\":\"successfully giving the final answer.\"}", 3
            else:
                "{error:\"\"return_type\" is not a valid choice\"}", 2
        else:

            for k, function in enumerate(self.functions):
                if function["name"].endswith(action_name):
                    pure_api_name = self.api_name_reflect[function["name"]]
                    # response = get_rapidapi_response(payload)
                    try:

                        payload = {
                            "category": self.cate_names[k],
                            "tool_name": self.tool_names[k],
                            "api_name": pure_api_name,
                            "tool_input": action_input,
                            "strip": "truncate",
                        }
                        # print(payload)
                        if self.process_id == 0:
                            print(colored(f"query to {self.cate_names[k]}-->{self.tool_names[k]}-->{action_name}",color="yellow"))
                        url  = "http://47.251.13.204:8080/rapidapi"
                        response = requests.post(url, json=payload,timeout=30)

                        if response.status_code != 200:
                            return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
                            
                        try:
                            response = response.json()
                        except:
                            print(response)
                            return json.dumps({"error": f"request invalid, data error", "response": ""}), 12
                        # 1幻觉函数名
                        # 4代表模型自己决定剪枝
                        # 5代表api调用timeout
                        # 6代表404
                        # 7代表未订阅
                        # 8代表unauthorized
                        # 9代表too many requests
                        # 10代表rate limit per minute
                        # 11信息包含 "error"字段
                        # 12,请求返回错误，500
                        if response["error"] == "API not working error...":
                            status_code = 6
                        elif response["error"] == "Unauthorized error...":
                            status_code = 7
                        elif response["error"] == "Unsubscribed error...":
                            status_code = 8
                        elif response["error"] == "Too many requests error...":
                            status_code = 9
                        elif response["error"] == "Rate limit per minute error...":
                            print("Reach api calling limit per minute, sleeping...")
                            time.sleep(60)
                            status_code = 10
                        elif response["error"] == "Message error...":
                            status_code = 11
                        else:
                            status_code = 0
                        return json.dumps(response), status_code # 这里看是否要截取observation
                    except Exception as e:
                        return json.dumps({"error": f"Timeout error...{e}", "response": ""}), 5
            return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1


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
    if method.startswith("CoT"): #单次模拟的UCT就是CoT
        passat = int(method.split("@")[-1])
        chain = single_chain(llm=llm_forward, io_func=env,process_id=process_id)
        result = chain.start(
                            pass_at=passat,
                            single_chain_max_step=single_chain_max_step,
                            max_query_count=max_query_count,
                            answer=5)
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
                            answer=5,
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
        pattern = r"ETS(.*)_s(\d+)_f(\d+)_t([\d\.]+)_p([\d\.]+)_m(\d+)_rn(\d+)_rg(\d+)"
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
            matching_interval = int(re_result.group(6))
            new_candidate_race_count = int(re_result.group(7))
            global_race_count = int(re_result.group(8))
            chain = ETS_tree_search(llm=llm_forward, io_func=env,process_id=process_id)
            result = chain.start(simulation_count=simulation_count,
                                p_new_node=p_new_node,
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


def run(method, backbone_model, query_id, data_dict, query, output_dir_path,tool_des,process_id=0):
    splits = output_dir_path.split("/")
    os.makedirs("/".join(splits[:-1]),exist_ok=True)
    os.makedirs("/".join(splits),exist_ok=True)

    output_file_path = os.path.join(output_dir_path,f"{query_id}_{backbone_model}_{method}.json")
    if os.path.exists(output_file_path):
        return

    env = wrap_rapidapi_multicate_multitool(data_dict, query,tool_descriptions=tool_des,process_id=process_id)

    if process_id == 0:
        print(colored(f"[process({process_id})]now playing {query}, with {len(env.functions)} APIs", "green"))
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

    return result


def contain(candidate_list,white_list):
    '''
    保证candidate_list里的所有东西都在white_list里
    '''
    output = []
    for cand in candidate_list:
        if cand not in white_list.keys():
            return False
        output.append(white_list[cand])
    return output


tool2query_count_hash = {}

def contain_at_least_once(candidate_list,white_list):
    '''
    保证candidate_list里最少一个在white_list里
    '''
    for cand in candidate_list:
        if cand in white_list.keys():
            if cand not in tool2query_count_hash.keys():
                tool2query_count_hash[cand] = 0
            tool2query_count_hash[cand] += 1
            if tool2query_count_hash[cand] < 10:
                return True
    return False

def main(query_dir, answer_dir, method,backbone_model):
    # 每个工具只先跑一条query
    white_list_cache_file = os.path.join(DATA_DIR,"white_list_new.pk")
    if os.path.exists(white_list_cache_file):
        white_list = pickle.load(open(white_list_cache_file,"rb"))
    else:
        white_list = get_white_list("jsons_filtered_pipeline_subscribed_white_list")
        pickle.dump(white_list,open(white_list_cache_file,"wb"))
    white_list_cache_file = os.path.join(DATA_DIR,"white_list_for_tool_generalize.pk")
    if os.path.exists(white_list_cache_file):
        white_list_small = pickle.load(open(white_list_cache_file,"rb"))
    else:
        white_list_small = get_white_list("jsons_filtered_pipeline_subscribed_white_list_multi_queries")
        pickle.dump(white_list_small,open(white_list_cache_file,"wb"))
    task_list = []
    with open(query_dir,"r") as reader:
        for k,line in tqdm(enumerate(reader)):
            
            try:
                data_dict = json.loads(line.strip()[:-1])
            except Exception as e:
                print(e)
                continue

            origin_tool_names = [cont["tool_name"] for cont in data_dict["api_list"]]



            tool_des = contain(origin_tool_names,white_list)
            if tool_des == False:
                continue
            if not contain_at_least_once(origin_tool_names,white_list_small): #包含的工具最少有一个在泛化工具集上
                continue
            tool_des = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]

            query = data_dict["query"]

            task_list.append((method, backbone_model, k, data_dict, query, answer_dir,tool_des))

    return task_list

def get_white_list(name):
    white_list_dir = os.path.join(DATA_DIR,name)
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
    answer_file_name = "answer_G2_singleanswer"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_query_file', type=str, default=os.path.join(DATA_DIR, "queryG2.json"), required=False, help='input path')
    parser.add_argument('--output_answer_file', type=str, default=os.path.join(DATA_DIR,answer_file_name), 
    required=False, help='output path')

    # parser.add_argument('--input_query_file', type=str, default=os.path.join(DATA_DIR,"queryG3.json"), required=False, help='input path')
    # parser.add_argument('--output_answer_file', type=str, default=os.path.join(DATA_DIR,"multicate_multitool_multianswer_new_format"), 
    # required=False, help='output path')

    parser.add_argument('--process_num', type=int, default=60, required=False, help='process number')
    parser.add_argument('--debug', type=int, default=0, required=False, help='1 for debugging')
    parser.add_argument('--backbone_model', type=str, default="ChatGPT", required=False, help='backbone_model')

    parser.add_argument('--method', type=str, default="DFS_woFilter_w2", required=False, help='method for answer generation: CoT@n,Reflexion@n,BFS_wn_en, DFS_woFilter_wn,UCT_vote, ETS_sn_fn_tn_pn_mn_rnn_rgn') 
    # ETS_annealing_sqrt_woInitElo_s10_f1_t173.72_p0.5_m3_rn3_rg4
    # ETS_random_woInitElo_s10_f1_t1.0_p0.5_m1_rn0_rg0
    args = parser.parse_args()

    query_dir = args.input_query_file
    answer_dir = args.output_answer_file
    method = args.method
    process_num = args.process_num
    debug = args.debug
    backbone_model = args.backbone_model


    task_list = main(query_dir, answer_dir, method,backbone_model)

    
    use_query_count = 5000000
    # subfix = str(use_query_count)
    subfix = "one_query_at_most_10"

    random.seed(42)
    random.shuffle(task_list)
    task_list = task_list[:use_query_count]

    print(f"total tasks: {len(task_list)}")

    new_task_list = []
    valid_query_id = []
    for task in task_list:
        out_dir_path = task[-2]
        query_id = task[2]
        valid_query_id.append(query_id)
        output_file_path = os.path.join(out_dir_path,f"{query_id}_{backbone_model}_{method}.json")
        if not os.path.exists(output_file_path):
            new_task_list.append(task)
    task_list = new_task_list

    with open(os.path.join(DATA_DIR,"valid_query_id",f"{answer_file_name}_{subfix}.json"),"w") as writer:
        json.dump(valid_query_id,writer)

    print(f"undo tasks: {len(task_list)}")


    q=queue.Queue(10000000)  #创建一个先进先出的队列
    starting_time = time.time()
    for task in task_list: #共享的任务列表，所有线程一起消费
        q.put(task)

    for i in range(process_num):
        p = Consumer(process_id=i,starting_time=starting_time)
        p.start()

