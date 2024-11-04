
from Downstream_tasks.base_env import base_env

from termcolor import colored
from run_webshop.env_utils import get_buy_info_from_json

import json
import os
import re
import traceback
import requests
import inspect
from RestGPT.test import env_datasets_path
# env_datasets_path = "RestGPT/datasets/tmdb.json"

task_description = f'''
You are going to use a lot of Rest API to solve tasks.
REMEMBER: ALWAYS give function call from ['CheckAPIDocs', 'Request', "Finish"], to give your action.
'''

class my_post:
    def do_post(url, **kwargs):
        # print(colored(f"post url: {url}", color='green'))
        req = kwargs
        req_json = json.dumps(req)
        res = requests.post(url, data=req_json.encode('utf-8'))
        if res.status_code != 200:
            print('post error: ', res.content)
            raise ValueError(f"error: {res.content}") 
        res_json = res.json()
        return res_json

    def my_rpc(return_list: list):
        def func_wrapper(func):
            def exec_wrapper(*args, **kwargs):
                try:
                    sign = inspect.signature(func)
                    param_name = sign.parameters.items()
                    param = {item[0]: args[i] if i < len(args) else kwargs[item[0]] for i,item in enumerate(param_name)}
                    # print(param['self'])
                    url = param['self'].url
                    param.pop('self', None)
                    res = my_post.do_post(url=url, function=func.__name__, **param)
                    result = [res[key] for key in return_list]
                    return tuple(result)
                except BaseException as e:
                    print(traceback.format_exc(), e)
                    raise e
            return exec_wrapper
        return func_wrapper

# addr = 'localhost'
# port = '12345'
# url = f'http://{addr}:{port}/api'
# my_post.url = url
# @my_post.my_rpc(return_list=['hello'])
# def say_hello(message):
#     pass

# print(say_hello('hello, server!'))

def format_env_result(observation, available_actions):
    prompt = ""
    max_len = 4096
    if len(observation) > max_len:
        observation = observation[:int(max_len-100)] + '...' + observation[int(-100):]
    prompt += observation
    return prompt

def format_step_info(observation, available_actions, action):
    return {"observation": observation, "available_actions": available_actions, "action": action}

def get_action_list(history):
    return [step_info['action'] for step_info in history]

class restbench_env(base_env):

    def __init__(self, goal_idx, output_dir_path, process_id=0):
        port_list = [12348]
        self.addr = 'localhost'
        self.port = port_list[(process_id) % len(port_list)]
        self.url = f'http://{self.addr}:{self.port}/api'
        # my_post.url = url
        self.idx = int(goal_idx)
        self.output_dir_path = output_dir_path
        self.process_id = process_id
        self.task_description = task_description
        self.tool_names = ['CheckAPIDocs', 'Request', "Finish"]
        self.check_api_docs_func = {
            "name": "CheckAPIDocs",
            "description": "You can use this function to check the docs of a specific api. For example: `GET /search/person`",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_name": {
                        "type": "string",
                        "description": "The api name whose documentation you want to get. For example, `GET /search/person`."
                    }
                },
                "required": ["api_name"],
            }
        }
        self.request_func = {
            "name": "Request",
            "description": "You can use this function to make rest api request. Please check the corresponding docs and follow the instruction in docs before you make this request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "The request method, GET, POST, etc."
                    },
                    "payload": {
                        "type": "string",
                        "description": "the payload of your request. Should be a string which can be parsed as a valid json."
                    }
                },
                "required": ["payload"],
            } 
        }
        self.finish_func = {
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
        
        self.functions = [self.check_api_docs_func, self.request_func, self.finish_func]
        observation, _, _, _, available_actions = (self.exec_action(self.idx, action_list=[]))
        self.input_description = format_env_result(observation, available_actions)
        self.reward = 0
        self.history = []
        self.done = False
        self.done_prompt = "You have succeeded to give your answer to the query. Task is completed. "
        self.success = 0
    
    def post(self, *args, **kwargs):
        return my_post.my_rpc(self.url, *args, **kwargs)

    def restart(self):
        '''
        
       '''
        pass
    
    def get_score(self):
        '''
        
        oracle()
        '''
        return self.reward

    def step(self,**args):
        '''
        
        '''
        obs, code = self._step(**args)
        # if len(obs) > 2048:
        #     obs = obs[:2048] + "..."
        return obs, code

    def _step(self, action_name="", action_input=""):
        '''
        action_name normalize
        observation string
        0
        1api
        2
        3final answer
        4
        '''
        # print(f"action history: {get_action_list(self.history)}")
        try:
            action_json = json.loads(action_input)
            if action_name == 'CheckAPIDocs':
                result = self.check_api_docs(action_json["api_name"])
                return result
            elif action_name == 'Request':
                result = self.request(method=action_json['method'], payload=action_json['payload'])
                return result
            elif action_name == "Finish":
                result = self.finish(action_json)
                return result
            else:
                return json.dumps({"error": f"No such function name: {action_name}"}), 1
        except BaseException as e:
            print(traceback.format_exc())
            with open(os.path.join(self.output_dir_path, "error.txt"), "a") as err_f:
                err_f.write(f"task {self.idx}\n history: {self.history}" + traceback.format_exc() + '\n' + str(e) + '\n')
            return f'{{"error": invalid json({e})}}', 2
    
    def check_api_docs(self, api_name):
        observation, available_actions = self.step_env({"action": "check_api_docs", "api_name": api_name})
        # self.set_select_list(available_actions)
        message = format_env_result(observation, available_actions), 0
        return message

    def request(self, method, payload):
        data = ""
        try:    
            data = json.loads(payload)
            observation, available_actions = self.step_env({"action": "request", 'method': method, "payload": data})
            message = format_env_result(observation, available_actions), 0
        except BaseException as e:
            print(traceback.format_exc(), str(e))
            message = f"payload error: {str(e)}", 2
        return message


    def finish(self, action_input):
        with open(os.path.join(env_datasets_path), "r", encoding='utf-8') as fr:
            json_data = json.load(fr)
            query = json_data[self.idx]['query']
        total_history = {
            "idx": self.idx,
            "history": self.history,
            "reward": self.reward,
            "result": action_input,
            "query": query
        }
        if not os.path.exists(self.output_dir_path):
            os.mkdir(self.output_dir_path)
        with open(os.path.join(self.output_dir_path,"restbench.jsonl"), 'a') as fw:
            data = json.dumps(total_history)
            fw.write(data + '\n')
        message = f"reward: {self.reward}"
        print(colored(message, color='cyan'))
        self.success = 1
        result = f"{self.done_prompt}\n You finished this task."
        print(colored(result, color='cyan'))
        return result, 3
    
    @my_post.my_rpc(return_list=["observation", "reward", "done", "info", "available_actions"])
    def exec_action(self, idx, action_list):
        '''
        action(observation, available_actions)
        '''
        pass

    def step_env(self, act):
        step_info = format_step_info(None, None, act)
        self.history.append(step_info)
        observation, self.reward, self.done, self.info, available_actions = self.exec_action(self.idx, [act])
        self.history[-1]["observation"] = observation
        self.history[-1]["reward"] = self.reward
        self.history[-1]["done"] = self.done
        return observation, available_actions

    def check_success(self):
        '''
        10
        '''
        return self.success

    def to_json(self):
        '''
        
        '''
        pass

