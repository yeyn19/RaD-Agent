from LLM.fake_llm import chatGPT
from LLM.openai_0613 import chatgpt_0613
from termcolor import colored
from Algorithms.single_chain import single_chain
from Algorithms.reflexion import reflexion_chain
from Algorithms.BFS import BFS_tree_search
from Algorithms.UCT_abstract import UCT_abstract
from Algorithms.UCT_vote import UCT_vote
from Algorithms.UCT_vote_function import UCT_vote_function
from Downstream_tasks.base_env import base_env
from utils import do_24
import re
import os
from functools import partial
import json

from typing import overload

from Downstream_tasks.tool_nolc import load_single_tools,import_all_apis

from pprint import pprint
import pdb

class wrap_wolframalpha(base_env):
    def __init__(self,input_description):
        super(wrap_wolframalpha, self).__init__()

        self.input_description = input_description

        tool_name, tool_url = 'wolframalpha',  "http://127.0.0.1:8079/tools/wolframalpha/"
        tool_name, tool_config = load_single_tools(tool_name, tool_url)

        self.tool, _ = import_all_apis(tool_config)
        self.name_to_tool_map = {tool.name: tool for tool in self.tool}

        self.tool_names = [tool.name for tool in self.tool]

        self.task_description = "The followings is the description of the valid actions:\n"

        for tool in self.tool:
            self.task_description += tool.name + tool.description + "\n"

        pattern_1 = re.compile(r"Your input should be a json \(args json schema\): {(.*)} The Action")
        pattern_2 = re.compile(r"{\"(.*)\" : (.*), }")
        for api in self.tool:
            func_name = api.name
            func_description = api.description

            function = {
                "name": func_name,
                "description": func_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
            }

            param_string = pattern_1.findall(func_description)

            for i in param_string:
                result = pattern_2.findall(i)[0]
                param_name, param_type = result[0], result[1]

                param = {
                    param_name: {
                        "type": param_type,
                        "description": "", # TODO
                    }
                }

                function["parameters"]["properties"].update(param)

            self.functions.append(function)

        self.finish_func = {
            "name": "Finish",
            "description": "If you think you get the result which can answer the input description, call this function to give the final answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer you want to give the user"
                    },
                },
                "required": ["answer"]
            }
        }
        self.functions.append(self.finish_func)
        self.tool_names.append(self.finish_func["name"])

        self.restart()

    def check_success(self):
        return self.status

    def to_json(self):
        return {}

    def restart(self):
        self.status = 0

    def get_score(self):
        return 0.0


    def step(self, action_name="",action_input=""):
        # print(action_input)

        if action_name == "Final Answer":
            self.status = 1
            return

        if self.name_to_tool_map.get(action_name):
            observation = self.name_to_tool_map[action_name].run(
                action_input
            )
            # print(observation)
            return observation, 0
        else:
            output = f"no such action : {action_name}"
            # print(output)
            return output, 0

def node_to_chain(node):
    chain = {
        "prompt": "",
        "query": "",
        "chains": [],
        "answer": "",
    }

    step = {}
    for i, message in enumerate(node.messages):
        if message["role"] == "system":
            chain["prompt"] = message["content"]
        elif message["role"] == "user":
            chain["query"] += message["content"]
        elif message["role"] == "assistant":
            if "function_call" in message:
                function_call = message["function_call"]
                if function_call["name"] == "Finish":
                    chain["answer"] = function_call["arguments"]
                else:
                    step["action"] = function_call["name"]
                    step["action_input"] = function_call["arguments"]
            else:
                step["thought"] = message["content"]
        elif message["role"] == "function":
            step["observation"] = message["content"]
            chain["chains"].append(step)
            step = {}

    return chain



def test_all(method):
    file_name = r"./Downstream_tasks/raw_data/17K_wolfram_processed.jsonl"
    output_dir = rf"./test_result/17K_wolfram_{method}"
    os.makedirs(output_dir,exist_ok=True)
    data_list = []
    with open(file_name,"r",encoding="utf-8") as f:
        data_list = json.load(f)
    
    total = 0
    win = 0
    for k, data in enumerate(data_list):
        total += 1

        flatten_input = str(k)
        if os.path.exists(os.path.join(output_dir,f"{flatten_input}.json")):
            with open(os.path.join(output_dir,f"{flatten_input}.json"),"r",encoding="utf-8") as reader:
                json_obj = json.load(reader)
                if json_obj["win"] == True:
                    win += 1
            continue


        env, chain = test_single(data["query"],method)
        if chain.status == 1:
            win += 1
            print(colored(f"You Win, ratio={win}/{total}={win/total}","green"))
        else:
            print(colored(f"You Lose, ratio={win}/{total}={win/total}","green"))

        with open(os.path.join(output_dir,f"{flatten_input}.json"),"w",encoding="utf-8") as writer:
            json_obj = chain.to_json()
            json.dump(json_obj,writer,indent=2)

def test_single(input_description):
    print(colored(f"now playing {input_description}", "green"))
    env = wrap_wolframalpha(input_description)

    llm_forward = chatgpt_0613()
    chain = UCT_vote_function(llm=llm_forward,io_func=env)
    result = chain.start(simulation_count=3,epsilon_new_node=0.3,choice_count=3,vote_candidates=2,vote_count=1,single_chain_max_step=40)
    chain = node_to_chain(result)
    pdb.set_trace()
    return env, chain


if __name__ == "__main__":
    task = '''Let $$x^8 + 3x^4 - 4 = p_1(x)p_2(x)\cdots p_k(x)$$, Where each non-constant polynomial p_i(x) is monic with integer coefficients, and cannot be factored further over the integers. Compute $$ p_1(1) + p_2(1) + \cdots + p_k(1) $$'''
    # task = "Convert $\\sqrt{2} e^{11 \\pi i/4}$ to rectangular form.\n"

    test_single(task)
    # test_all("CoT")