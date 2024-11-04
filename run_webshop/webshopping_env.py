
from Downstream_tasks.base_env import base_env
from AgentBench.webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

from termcolor import colored
import json
import os
import re
import traceback
# 1. stepenv
# 2. buy now
# 3. 10openai request
# 4. Prompt

# ['back to search', '< prev', 'description', 'features', 'reviews', 'buy now', 'blue-dinosaur', 'blue-donut', 'pink-dinosaur', 'pink-donut', 'white-dinosaur']


def show_select_list(available_actions):
    print(available_actions)
    select_list = available_actions["clickables"][:]
    remove_list = ['back to search', 'next >', '< prev', 'search']
    for rem in remove_list:
        if rem in select_list:
            select_list.remove(rem)
    return select_list[:]

def show_buy_list(available_actions):
    buy_list = available_actions["clickables"][:]
    remove_list = ['back to search', 'next >', '< prev', 'description', 'features', 'reviews', 'buy now']
    for rem in remove_list:
        if rem in buy_list:
            buy_list.remove(rem)
    return buy_list[:]

def extract_query(string):
    pattern = re.compile(r"\[SEP\] (.*?) \[SEP\]")
    match = pattern.search(string)
    if match:
            result = match.group(1)
            return result
    else:
        return ""
def extract_content(string):
    pattern = re.compile(r"< Prev \[SEP\] (.*)")
    match = pattern.search(string)
    if match:
        result = match.group(1)
        return result
    else:
        return ""

def format_env_result(observation, available_actions):
    page_name = 'Product Details Page' if 'buy now' in str(available_actions) else 'Product Selection Page'
    prompt = f'You are now at {page_name}. \n[PAGE_BEGIN]\n'
    max_len = 2048
    if len(observation) > max_len:
        observation = observation[:max_len]
    prompt = prompt + observation + "\n[PAGE_END]\n"
    prompt = prompt + "Now you can do: \n"
    if page_name == 'Product Details Page':
        prompt = prompt + f'1. buy: {show_buy_list(available_actions)}\n2. search\n'
    else:
        prompt = prompt + f'1. select: {show_select_list(available_actions)}\n2. search\n'
    return prompt

def format_step_info(observation, available_actions, action):
    return {"observation": observation, "available_actions": available_actions, "action": action}

def get_action_list(history):
    return [step_info['action'] for step_info in history]

class webshopping_env(base_env):

    env = WebAgentTextEnv(observation_mode="text", human_goals=True)

    def __init__(self, goal_idx: int, output_dir_path, process_id=0):
        self.idx = goal_idx
        self.output_dir_path = output_dir_path
        self.process_id = process_id
        self.task_description = f'''You should use functions to help handle the web shopping task on a web shop site.
We have 2 Pages: Product Selection Page & Product Details Page.
You have access of the following functions:
1. search: at any time, you can search a product by keywords. Then you will goto the Product Selection Page which shows a list of related products.
2. select: after search keywords, you can select a product at Product Selection Page. Then you will goto the Product Details Page, which shows the details of the product you select.
3. buy: at Product Details Page, you can buy a product. Then the shopping task is completed.
'''
        self.tool_names = ['search', 'select', 'buy']
        self.search_func = {
            "name": "search",
            "description": "You can use this function at ANY page(BOTH Product Selection Page and Product Details Page). If you want to use the search box to search products by keywords, you can call this function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "The keywords which will be used to search products you want."
                    }
                },
                "required": ["keywords"],
            }
        }
        self.select_func = {
           "name": "select",
            "description": "You can use this function at Product Selection Page after search keywords. Input the name of a product on the current shopping webpage, then you will goto the Product Details Page, which shows the details of the product you select.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product you want to select. You should choose from the select list."
                    }
                },
                "required": ["product"],
            } 
        }
        self.buy_func = {
            "name": "buy",
            "description": "You can use this function at Product Details Page. Call this function to buy the selected product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attribute": {
                        "type": "string",
                        "description": "If the product has different attributes, then you should choose a proper attribute here when you decide to buy the product. If no attributes in the list, you don't need to fill this parameter."
                    }
                },
                "required": [],
            }
        }
        # back_func = {
        #     "name": "back_to_search",
        #     "description" : "You can use this function at Product Details Page. Call this function, and you will go back to Product Selection Page. ", 
        #     "parameters": {

        #     }
        # }
        self.functions = [self.search_func, self.select_func, self.buy_func]
        self.input_description = format_env_result(*(self.exec_prev_actions(self.idx, action_list=[])))
        self.reward = 0
        self.history = []
        self.done = False
        self.done_prompt = "You have succeeded to buy this product. Task is completed. "
        self.success = 0
    def restart(self):
        '''
        
       '''
        webshopping_env.env.reset(self.inputs[self.idx])
    
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
        print(f"action history: {get_action_list(self.history)}")
        try:
            action_json = json.loads(action_input)
            if action_name == 'search':
                result = self.search(action_json["keywords"])
                return result
            elif action_name == 'select':
                result = self.select(action_json['product'])
                return result
            elif action_name == 'buy':
                if 'attribute' in action_json.keys():
                    result = self.buy(action_json['attribute'])
                else:
                    result = self.buy()
                return result
            else:
                return json.dumps({"error": f"No such function name: {action_name}"}), 1
        except BaseException as e:
            print(traceback.format_exc())
            with open(os.path.join(self.output_dir_path, "error.txt"), "a") as err_f:
                err_f.write(f"task {self.idx}\n history: {self.history}" + traceback.format_exc() + '\n' + str(e) + '\n')
            return f'{{"error": invalid json({e})}}', 2
    
    def search(self, keywords):
        observation, available_actions = self.exec_prev_actions(self.idx, get_action_list(self.history))
        act = f"search[{keywords}]"
        if not available_actions['has_search_bar']:
            observation, available_actions = self.step_env(observation, available_actions, "click[back to search]")
        observation, available_actions = self.step_env(observation, available_actions, act)
        available_actions = webshopping_env.env.get_available_actions()

        self.set_select_list(available_actions)
        message = format_env_result(observation, available_actions), 0
        return message

    def select(self, product):
        observation, available_actions = self.exec_prev_actions(self.idx, get_action_list(self.history))
        product = product.lower()
        act = f"click[{product}]"
        if 'buy now' in str(available_actions):
            message = "Error: You can't select product at this page.", 2
        elif product in [str(action) for action in available_actions['clickables']]:
            act = f"click[{product}]"
            observation, available_actions = self.step_env(observation, available_actions, act)

            self.set_buy_list(available_actions)

            all_obs = observation
            observation, available_actions = self.step_env(observation, available_actions, 'click[description]')
            all_obs += "\ndescription: \n" + extract_content(observation)
            observation, available_actions = self.step_env(observation, available_actions, 'click[< prev]')
            observation, available_actions = self.step_env(observation, available_actions, 'click[features]')
            all_obs += "\nfeatures: \n" + extract_content(observation)
            observation, available_actions = self.step_env(observation, available_actions, 'click[< prev]')
            observation, available_actions = self.step_env(observation, available_actions, 'click[reviews]')
            all_obs += "\nreviews: \n" + extract_content(observation)
            observation, available_actions = self.step_env(observation, available_actions, 'click[< prev]')
            message = format_env_result(all_obs, available_actions), 0
        else:
            message = "Error: The product doesn't exist.", 2
        return message
    def buy(self, attribute=None):
        observation, available_actions = self.exec_prev_actions(self.idx, get_action_list(self.history))
        if 'buy now' in str(available_actions):
            if attribute:
                if attribute in str(available_actions):
                    observation, available_actions = self.step_env(observation, available_actions, f"click[{attribute}]")    
                    observation, _ = self.step_env(observation, available_actions, "click[buy now]")
                    message = self.finish()
                else:
                    message = f"Error: attribute {attribute} doesn't exist.", 2
            else:
                observation, _ = self.step_env(observation, available_actions, "click[buy now]")
                message = self.finish() 
        else:
            message = "Error: You can't buy product at this page.", 2
        return message
    def finish(self):
        total_history = {
                "idx": self.idx,
                "history": self.history,
                "reward": self.reward
        }
        if not os.path.exists(self.output_dir_path):
            os.mkdir(self.output_dir_path)
        with open(os.path.join(self.output_dir_path,"webshop.jsonl"), 'a') as fw:
            data = json.dumps(total_history)
            fw.write(data + '\n')
        message = f"reward: {self.reward}"
        self.success = 1
        print(message)
        return self.done_prompt, 3
    def exec_prev_actions(self, idx,action_list):
        '''
        action(observation, available_actions)
        '''
        webshopping_env.env.reset(idx)
        observation, available_actions = (webshopping_env.env.observation, webshopping_env.env.get_available_actions())
        for action in action_list:
            print(f'perform action: {action}')
            observation, _, done, _ = webshopping_env.env.step(action)
            available_actions = webshopping_env.env.get_available_actions()
        return (observation, available_actions)

    def set_select_list(self, available_actions):

        self.select_func['parameters']['properties']['product']['enum'] = show_select_list(available_actions)
        # print("set select list: ", self.functions[1])

    def set_buy_list(self, available_actions):
        self.buy_func['parameters']['properties']['attribute']['enum'] = show_buy_list(available_actions)
        # print("set buy list: ", self.functions[2])

    def step_env(self, observation, available_actions, act):
        step_info = format_step_info(observation, available_actions, act)
        self.history.append(step_info)
        # check the validity of act
        if act.startswith("search"):
            if not available_actions['has_search_bar']:
                print(colored(f"Failed to {act}!!! Available: {available_actions}", color="magenta"))
        elif act.startswith('click'):
            clickable_name = act[6:-1]
            # print('clickable_name', clickable_name)
            if clickable_name not in [str(action) for action in available_actions['clickables']]:
                print(colored(f"Failed to {act}!!!Available: {available_actions}", color="magenta"))
        observation, self.reward, self.done, self.info = webshopping_env.env.step(act)
        self.history[-1]["reward"] = self.reward
        self.history[-1]["done"] = self.done
        avaliable_actions = webshopping_env.env.get_available_actions()
        return observation, avaliable_actions

    def check_success(self):
        '''
        10
        '''
        return self.success

    def to_json(self):
        '''
        
        '''
        webshopping_env.env.reset(self.idx)
