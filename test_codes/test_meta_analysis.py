from LLM.fake_llm import chatGPT
from termcolor import colored
from Algorithms.single_chain import single_chain
from Algorithms.reflexion import reflexion_chain
from Algorithms.BFS import BFS_tree_search
from Algorithms.UCT_abstract import UCT_abstract
from Algorithms.UCT_vote import UCT_vote
from Downstream_tasks.base_env import base_env
from utils import do_24
import re
import os
from functools import partial
import json

from typing import overload

from Downstream_tasks.tool_nolc import load_single_tools,import_all_apis
from langchain.schema import AgentAction, AgentFinish


class wrap_meta_analysis(base_env):
    def __init__(self,input_description):
        super(wrap_meta_analysis, self).__init__()

        self.task_description = '''The followings is the description of the valid actions:
1.search_literature: search for the given topic literatures in the database and return the path of literatures file and the number of literatures. the searching term should be key words in the topic (2-5 words). the number of literatures will be less than maxnum (recommend 30). Your input should be a json (args json schema): {\"topic\" : string, \"maxnum\" : integer, \"term\" : string, } The Action to trigger this API should be search_literature and the input parameters should be a json dict string. Pay attention to the type of parameters.
2.split_criteria: split the screening requirements in the criteria of the literatures into a series of simple yes/no problems, and return the path of the splitted questions.. Your input should be a json (args json schema): {\"criteria\" : string, } The Action to trigger this API should be split_criteria and the input parameters should be a json dict string. Pay attention to the type of parameters.
3.literature_filter: Check each literatures saved in the literature path according to the questions saved in the question path, and return the literatures that match the requirements. Concat path is the concatenated string of literature path and question path connected with '&&&'.. Your input should be a json (args json schema): {\"concat_path\" : string, } The Action to trigger this API should be literature_filter and the input parameters should be a json dict string. Pay attention to the type of parameters.
4.draw_table: extract the important elements of the literatures recorded in the literature path and return the path of table records. concatenate the literature path and the analysis topic with '&&&' as the input.. Your input should be a json (args json schema): {\"literature_path_and_topic\" : string, } The Action to trigger this API should be draw_table and the input parameters should be a json dict string. Pay attention to the type of parameters.
5.combine_table: combine several tables recorded in the table path into one comprehensive record table and return. give the literature path, table path and the exploring topic as the input.. Your input should be a json (args json schema): {\"literature_path\" : string, \"table_path\" : string, \"topic\" : string, } The Action to trigger this API should be combine_table and the input parameters should be a json dict string. Pay attention to the type of parameters.
6.generate_summary: given the exploring topic and the record table path of the literatures, this tool generates a paragraph of summary.. Your input should be a json (args json schema): {\"topic\" : string, \"table_path\" : string, } The Action to trigger this API should be generate_summary and the input parameters should be a json dict string. Pay attention to the type of parameters.
7.print_literature: given the literature path and number that are required to display, this tool returns the title and abstract of the literature.. Your input should be a json (args json schema): {\"literature_path\" : string, \"print_num\" : integer, } The Action to trigger this API should be print_literature and the input parameters should be a json dict string. Pay attention to the type of parameters.
8.print_tablefile: given the table file path that are required to display, this tool reads the file and returns the string of the table.. Your input should be a json (args json schema): {\"table_path\" : string, } The Action to trigger this API should be print_tablefile and the input parameters should be a json dict string. Pay attention to the type of parameters.
''' 

        self.input_description = "Your task is: "+ input_description

        tool_name, tool_url = 'meta_analysis',  "http://127.0.0.1:8079/tools/meta_analysis/"
        tool_name, tool_config = load_single_tools(tool_name, tool_url)

        self.tool = import_all_apis(tool_config)
        self.name_to_tool_map = {tool.name: tool for tool in self.tool}

        self.tool_names = [tool.name for tool in self.tool]
        # print(self.tool_names)

        self.restart()

    def check_success(self):
        return False

    def to_json(self):
        return {}

    def get_score(self):
        return 0.0

    def restart(self):
        self.status = 0


    def step(self, action_name="",action_input=""):
        # print(action_input)
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

            

def test_all(method):
    file_name = r"C:\git_repos\LLM-AlphaGo\Downstream_tasks\raw_data\2k_meta_analysis_single_processed.jsonl"
    output_dir = rf"C:\git_repos\LLM-AlphaGo\test_result\2k_meta_analysis_{method}"
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

def test_single(input_description,method):
    print(colored(f"now playing \"{input_description}\"","green"))
    env = wrap_meta_analysis(input_description)
    input_description = env.input_description
    # print(env.input_description)
    llm = chatGPT()
    llm_forward = partial(llm.generation,model="gpt-3.5-turbo")
    if method == "CoT":
        chain = single_chain(input_description=input_description,pass_at=1,llm=llm_forward,io_func=env)
        result = chain.start(single_chain_max_step=30)
    elif method == "Reflexion":
        chain = reflexion_chain(input_description=input_description, llm=llm_forward,io_func=env)
        result = chain.start(max_chain_count=3,single_chain_max_step=30)
    elif method == "BFS":
        chain = BFS_tree_search(input_description=input_description,llm=llm_forward,io_func=env)
        result = chain.start(search_width=3, expansion_ratio=3, single_chain_max_step=18,max_iters=4)
    elif method == "UCT_abstract":
        chain = UCT_abstract(input_description=input_description,llm=llm_forward,io_func=env)
        result = chain.start(simulation_count=10,single_chain_max_step=18)
    elif method == "UCT_vote":
        chain = UCT_vote(input_description=input_description,llm=llm_forward,io_func=env)
        result = chain.start(simulation_count=10,epsilon_new_node=0.3,choice_count=3,vote_candidates=2,vote_count=2,single_chain_max_step=18)
    return env, chain


if __name__ == "__main__":
    # test_single("Help me find studies that explore the relationship between caffeine consumption and sleep quality. Please print the titles and abstracts of the studies as well as generate a table that compares the results of different studies.","CoT")
    test_all("CoT")