import os
import json
from Downstream_tasks.base_env import base_env
from Algorithms.single_chain import single_chain
from Algorithms.reflexion import reflexion_chain
from Algorithms.BFS import BFS_tree_search
from Algorithms.UCT_abstract import UCT_abstract
from Algorithms.UCT_vote import UCT_vote
from Downstream_tasks.tool_nolc import load_single_tools,import_all_apis
from LLM.fake_llm import chatGPT
from termcolor import colored
import utils
from functools import partial


tool_description_prompt = '''Tool \"{tool_name}\" can help you do the following things:{tool_description}.
And you can interact with \"{tool_name}\" by the following apis:
{api_description_name}
'''

api_description_prompt = '''{api_order}.\"{api_name}\":{api_description}. {input_type_description}
'''
input_type_description_prompt = '''The valid input looks like {json_example}. Where {key_description}.
Pay attention to the type of parameters.
'''


class wrap_rapid_api(base_env):
    def __init__(self,js_data,category):
        super(wrap_rapid_api,self).__init__()
        # print(js_data)
        
        self.task_description = tool_description_prompt
        self.task_description = self.task_description.replace("{tool_name}",js_data["tool_name"])
        self.task_description = self.task_description.replace("{tool_description}",js_data["tool_description"])

        self.tool_names = []
        for k,api in enumerate(js_data["query_list"]):
    
            if api["method"] != "GET":
                continue
            api_description = api_description_prompt
            api_description = api_description.replace("{api_order}",str(k+1))
            api_description = api_description.replace("{api_name}",api["api_name"])
            api_description = api_description.replace("{api_description}",api["api_description"])
            self.tool_names.append(api["api_name"])

            if "required_parameters" in api.keys() and api["required_parameters"] != []:
                input_des = {}
                input_type_description = input_type_description_prompt
                key_description = ""

                for required_parameter in api["required_parameters"]:
                    input_des[required_parameter["name"]] = required_parameter["type"]
                    key_description += f"{required_parameter['name']} is a {required_parameter['type']}"
                    if required_parameter['description'] != "":
                        key_description += f", {required_parameter['description']}. "
                    key_description += "\n"

                input_type_description = input_type_description.replace("{json_example}",json.dumps(input_des))
                input_type_description = input_type_description.replace("{key_description}",key_description)


                api_description = api_description.replace("{input_type_description}",input_type_description)
            else:
                api_description = api_description.replace("{input_type_description}","")
            
            self.task_description = self.task_description.replace("{api_description_name}", api_description+"{api_description_name}")
        self.task_description = self.task_description.replace("{api_description_name}", "")
        # print(self.task_description)
        # exit()
        rapid_api_name = f"{utils.standardize(js_data['tool_name'])}_for_{category}"
        tool_name, tool_url = rapid_api_name,  f"http://127.0.0.1:8079/tools/{rapid_api_name}/"
        tool_name, tool_config = load_single_tools(tool_name, tool_url)

        self.tool = import_all_apis(tool_config)
        self.name_to_tool_map = {tool.name: tool for tool in self.tool}

        self.tool_names = [tool.name for tool in self.tool]

        self.restart()

    def check_success(self):
        return False

    def to_json(self):
        return {}

    def get_score(self):
        return 0.0

    def restart(self,input_description = ""):
        if input_description != "":
            self.input_description = input_description
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

    




def deal_single_rapid_api(path,category):

    with open(path,"r",encoding="utf-8") as reader:
        js_data = json.load(reader)
    env = wrap_rapid_api(js_data,category)

    llm = chatGPT()
    llm_forward = partial(llm.generation,model="gpt-3.5-turbo")

    for query in js_data["query_list"]:
        for nl_query in query["queries"]:
            print(colored(f"now playing \"{nl_query}\"","green"))
            input_description = f"Your task is: {nl_query}"
            env.restart(input_description)

            search_obj = single_chain(input_description = input_description,llm = llm_forward,io_func=env,pass_at=1)
            result = search_obj.start(single_chain_max_step=18)



if __name__ == "__main__":
    # data_root_path = r"C:\git_repos\LLM-AlphaGo\make_data\tool_queries"
    # output_root_path =  r"C:\git_repos\LLM-AlphaGo\make_data\tool_outputs"
    # for category in os.listdir(data_root_path):
    #     os.makedirs(os.path.join(output_root_path,category),exist_ok=True)
    #     for api in os.listdir(os.path.join(data_root_path,category)):
    #         deal_single_rapid_api(os.path.join(data_root_path,category,api))
    deal_single_rapid_api(r"C:\git_repos\LLM-AlphaGo\make_data\tool_queries\Gaming\elden_ring_wiki.json","Gaming")

