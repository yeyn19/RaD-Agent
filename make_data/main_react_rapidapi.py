import os
import json
import re
from utils import MyMRKLOutputParser
import requests
import yaml
import time
import jsonlines
from math import floor, ceil
import argparse
import multiprocessing as mp
from langchain import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from bmtools.agent.apitool import RequestTool
from bmtools.agent.executor import Executor
from bmtools.models.customllm import CustomLLM
import sys

def remove_blank(string):
    while "\n" in string or "    " in string:
        string = string.replace("\n", "")
        string = string.replace("    ", "")
    return string

def load_single_tools(tool_name, tool_url):
    
    # tool_name, tool_url = "datasette", "https://datasette.io/"
    # tool_name, tool_url = "klarna", "https://www.klarna.com/"
    # tool_name, tool_url =  'chemical-prop',  "http://127.0.0.1:8079/tools/chemical-prop/"
    # tool_name, tool_url =  'douban-film',  "http://127.0.0.1:8079/tools/douban-film/"
    # tool_name, tool_url =  'weather',  "http://127.0.0.1:8079/tools/weather/"
    # tool_name, tool_url =  'wikipedia',  "http://127.0.0.1:8079/tools/wikipedia/"
    # tool_name, tool_url =  'wolframalpha',  "http://127.0.0.1:8079/tools/wolframalpha/"
    # tool_name, tool_url =  'klarna',  "https://www.klarna.com/"


    get_url = tool_url +".well-known/ai-plugin.json"
    response = requests.get(get_url)

    if response.status_code == 200:
        tool_config_json = response.json()
    else:
        raise RuntimeError("Your URL of the tool is invalid.")

    return tool_name, tool_config_json

# basically copy the codes in singletool.py
def import_single_api(tool_json, tool_desc=""):
    '''import all apis that is a tool
    '''
    doc_url = tool_json['api']['url']
    response = requests.get(doc_url)

    if doc_url.endswith('yaml') or doc_url.endswith('yml'):
        plugin = yaml.safe_load(response.text)
    else:
        plugin = json.loads(response.text)

    server_url = plugin['servers'][0]['url']
    if server_url.startswith("/"):
        server_url = "http://127.0.0.1:8079" + server_url
    all_apis = []
    for key in plugin['paths']:
        value = plugin['paths'][key]
        try:
            api = RequestTool(root_url=server_url, func_url=key, method='get', request_info=value)
        except:
            continue
        if remove_blank(tool_desc).replace("{", "{{").replace("}", "}}") in remove_blank(api.description):
            all_apis.append(api)
            break # prompttoolapiqueryapi
    if len(all_apis) == 0:
        return None
    else:
        return all_apis

class STQuestionAnswerer:
    def __init__(self, stream_output=False, llm_model=None):
        self.llm = llm_model
        self.stream_output = stream_output

    def load_tools(self, name, meta_info, prompt_type="react-with-tool-description", return_intermediate_steps=True, selected_api_desc=""):

        self.all_tools_map = {}
        self.all_tools_map[name] = import_single_api(meta_info, selected_api_desc)
        if self.all_tools_map[name] == None:
            return None, None
        tool_str = "; ".join([t.name for t in self.all_tools_map[name]])
        prefix = f"""Answer the following questions as best you can. Specifically, you have access to the following APIs:"""
        suffix = """Begin! Remember: (1) Do not repeat the same action and action input again and again. (2) Follow the format, i.e,\nThought:\nAction:\nAction Input:\nObservation:\nFinal Answer:\n (3) Provide as much as useful information in your Final Answer. (4) Do not make up anything, and if your Observation has no link, DO NOT hallucihate one. (5) The action input should always be one json dict. Do not add additional marks such as "```" before of after the dict. (6) If you have enough information and want to stop the process, please use \nThought: I have got enough information\nFinal Answer: **your response. \n The Action: MUST be one of the following:""" + tool_str + """\nQuestion: {input}\n Agent scratchpad (history actions):\n {agent_scratchpad}"""
        format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times, max 7 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
        prompt = ZeroShotAgent.create_prompt(
            self.all_tools_map[name], 
            prefix=prefix, 
            suffix=suffix, 
            format_instructions=format_instructions,
            input_variables=["input", "agent_scratchpad"]
        )
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in self.all_tools_map[name] ]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        # parser
        agent.output_parser = MyMRKLOutputParser()
        if self.stream_output:
            agent_executor = Executor.from_agent_and_tools(agent=agent, tools=self.all_tools_map[name] , verbose=True, return_intermediate_steps=return_intermediate_steps)
        else:
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=self.all_tools_map[name], verbose=True, return_intermediate_steps=return_intermediate_steps)
        return agent_executor, prompt.template

if __name__=='__main__':

    tool_name = "weather"
    tool_url = f"http://127.0.0.1:8079/tools/{tool_name}/"
    tool_name, tools_config = load_single_tools(tool_name, tool_url)
    api_description = "Get today's the weather"
    qa =  STQuestionAnswerer()
    # llm
    customllm = CustomLLM()
    qa.llm = customllm
    agent, prompt_template = qa.load_tools(tool_name, tools_config, selected_api_desc=api_description)
    output = agent("Your query.")
