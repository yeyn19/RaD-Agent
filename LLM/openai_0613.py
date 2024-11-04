'''

'''
import ast
import concurrent
from csv import writer
import json
import math
import traceback
import openai
import os
import pandas as pd
import requests
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from tqdm import tqdm
from termcolor import colored
import time

from pprint import pprint
import pdb

# from LLM.pool import pool

api_key_pool = [
    "sk-1", #set your openai api key here
]

api_key = "sk-1" #set your openai api key here
base_url = "https://api.openai.com/v1/chat/completions"

def get_keys():
    orgs,keys = [],[]
    # lines = (pool_7).split("\n")
    for line in api_key_pool:
        if line.startswith('sk-'):
            keys.append(line)
        orgs.append(line)
        # elif "----" in line:
        #     conts = line[2:].split("----")
        # else:
        #     conts = line[2:].split("|")
        # for cont in conts:
        #     if cont.startswith("sk-"):
        #         keys.append(cont.strip())
        #     if cont.startswith("org-"):
        #         orgs.append(cont.strip())
    return orgs, keys
orgs3, keys3 = get_keys()
invalid_positions = []
orgs4 = []
keys4 = [api_key]

now_pos = -1

print(f"ChatGPT pool length: {len(keys3)}, valid pool: {len(keys3) - len(invalid_positions)}")

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(6))
def chat_completion_request(messages, functions=None,function_call=None,key_pos=None, model="gpt-3.5-turbo-16k-0613",stop=None,process_id=0, **args):
    global now_pos
    if now_pos == -1:
        now_pos = process_id*457 + math.floor(len(keys3) / 2)

    use_messages = []
    for message in messages:
        if not("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)
    model = "gpt-3.5-turbo"
    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 4096,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    json_data.update({"temperature": 1.0})
    if stop is not None:
        json_data.update({"stop": stop})
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    
    try:
        # Official OpenAI API
        now_pos = (now_pos + 1)%len(keys3) #
        while now_pos in invalid_positions:
            now_pos = (now_pos + 1)%len(keys3) #
        if key_pos == None:
            key_pos = now_pos
        # print(key_pos)
        openai.api_key = api_key
        openai.base_url = base_url
        try:
            
            openai_response = openai.chat.completions.create(
                **json_data,
                # request_timeout=120,
            )
        except BaseException as e:
            print(traceback.format_exc(), str(e))
            import pdb; pdb.set_trace()
        # print(openai_response)
        json_data = json.loads(str(openai_response))
        # with open(os.path.join("./success.txt"), "a") as fa:
        #     fa.write(keys3[key_pos] + '\n')
        return json_data 

    except Exception as e:
        traceback.print_exc()
        print("Unable to generate ChatCompletion response")
        print(f"Model: {model}, OpenAI calling Exception: {e}")
        return e

class chatgpt_0613:
    def __init__(self, model="gpt-3.5-turbo-16k-0613"):
        self.conversation_history = []
        self.time = time.time()
        self.TRY_TIME = 6
    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self,functions,process_id,key_pos=None,**args):
        
        # while time.time() - self.time < 3: #3s
        #     continue
        self.time = time.time()
        
        # function name64
        func_name_map = {}
        for function_dict in functions:
            function_name = function_dict["name"]
            if len(function_name) >= 64:
                cut_func_name = function_name[:64] #api_for_tool
                func_name_map[cut_func_name] = function_name
                function_dict["name"] = cut_func_name
            else:
                func_name_map[function_name] = function_name
        conversation_history = self.conversation_history
        json_data = None
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            if functions != []:
                json_data = chat_completion_request(
                    conversation_history, functions=functions,process_id=process_id, key_pos=key_pos,**args
                )
            else:
                json_data = chat_completion_request(
                    conversation_history,process_id=process_id,key_pos=key_pos, **args
                )
            try:
                total_tokens = json_data['usage']['total_tokens']
                message = json_data["choices"][0]["message"]
                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {json_data['usage']['total_tokens']}")

                if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]

                return message, 0, total_tokens
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
    
        try:
            output = str(json_data)
        except:
            output = "Unknown LLM output error"
        return {"role": "assistant", "content": output}, -1, 0



def ping_all_api_key():
    for i in range(len(keys3)):
        if i in invalid_positions:
            continue
        print(f"key_pos={i}")
        llm = chatgpt_0613(model="gpt-3.5-turbo-16k-0613") # gpt-4-32k-0613 gpt-3.5-turbo-16k-0613
        llm.TRY_TIME=1
        prompt = '''hello_chatGPT'''
        messages = [
            {"role":"system","content":""},
            {"role":"user","content":prompt},
        ]
        llm.change_messages(messages)
        output,error_code,total_tokens = llm.parse(functions=[],process_id=0,key_pos=i)
        print(output)
