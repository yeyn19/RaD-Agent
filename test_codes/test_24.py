from LLM.fake_llm import chatGPT
from LLM.openai_0613 import chatgpt_0613
from termcolor import colored
from Algorithms.single_chain import single_chain
from Algorithms.reflexion import reflexion_chain
from Algorithms.BFS import BFS_tree_search
from Algorithms.UCT_abstract import UCT_abstract
from Algorithms.UCT_vote import UCT_vote
from Algorithms.UCT_vote_function import UCT_vote_function
from Algorithms.ETS import ETS_tree_search
from Downstream_tasks.base_env import base_env
from utils import do_24
import re
import os
from functools import partial
import json

from typing import overload

example_1 = '''
Action: play_24
Action Input: x operations y = result, (left: a b c)
Where x,y is the number you want to combnine now, and the Observation will tell you what numbers are left now.
Because there are 4 numbers, so you only need 3 actions to combine all numbers, you cannot undo your actions.
Here is one examples: [3,13,9,7]
Action: play_24
Action Input: 3*13=39 (left: 39 9 7)
Observation: 3*13=39, left: [9,7,39]
Thought: There left 3 numbers 9,7,39, they are both not the factor of 24. So the final operation must be + or -. If I use 39 in last step, the other value need to be 24+39=63, and 63 equals to 7*9. I will combine 7 and 9.
Action: play_24
Action Input: 7*9=63 (left: 39 63)
Observation: 7*9=63, left: [63,39]
Thought: There are only 2 numbers 63,39 now. 63+39=102, 63*39>24, 63/39 is not a integer, 63-39=24. I make it!
Action: play_24
Action Input: 63-39=24 (left: 24)
Observation: 63-39=24, left: [24], you win
'''

example_2 = '''
<example_2>: [1,4,9,11]
Thought: There is a 1 in numbers, This number will not affect the result if it multiply other numbers. So look at other number. They seem to add to 24.
Action: play_24
Action Input: 4+9=13 (left: 1 13 11)
Observation: 4+9=13, left numbers: [1,4,9,11] -> [1,11,13]
Thought: There are 1,11,13 left. 1 will not affect the result if it multiply other numbers. I will try to get 24 with 11 and 13. 11*13>24, 11/13 is not a integer. But 11+13=24, I get it.
Action: play_24
Action Input: 11+13=24 (left: 1 24)
Observation: 11+13=24, left numbers: [1,11,13] -> [1,24]
Action: play_24
Action Input: 1*24=24 (left: 24)
Observation: 1*24=24, left numbers: [1,24] -> [24], you win
'''

# Here is some chain of thought examples:
# 1.assert your input is [1,1,4,6]: first you find 1 and 0 is trival to this task, because *1 and +0 is no influence, then you find the remain numbers 4,6 can combine by 4*6=24. So the final answer is 1*1*4*6=24.
# 2.assert your input is [1,4,9,11]: You may first find 4+9+11=24, and 1 is no influence. You then make it.
# 3.assert your input is [3,13,9,7]: This input is very hard, because none of input numbers are factor of 24, so maybe last step is + or -. Maybe the difference 3*13 and 9*7 is close to 24, you can try 7*9-3*13.
# 4.assert your input is [4,6,7,8]: First you find 4*6=24, but then it's hard to decrease other numbers. Then you find 4 is factor of 24, but can you conbine 6,7,8 to 6=(24/4) or 18=(24-6) or 30=24+6, it's hard. But 8 is also factor of 24. you can conbine 4,6,7 with 4 - (7-6) = 3. and 3*8=24.

class play_24(base_env):
    def __init__(self,data_list):
        super(play_24, self).__init__()
        self.task_description = f'''Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number 24. Each step, you are only allowed to choose two of the left numbers to obtain a new number. For example, 7*9 - 3*13 = 24.
Remember:
1.all of the number must be used , and must be used ONCE. So Only when left numbers is exact 24, you will win. So you don't succeed when left number = [24, 5]. You succeed when left number = [24]. 
2.all the try takes exactly 3 steps, and the task ends when the count of left numbers if 1, and check whether the only left number equals 24.
3.If there are only 2 left numbers, you can try to use all the operations first(+-*/), and find if there is a way to combine to 24. For example: left number [3,7], you trys 3+7=7+3=10, 3-7=-4, 7-3=4, 3*7=7*3=21, 3/7 = 0.428, 7/3=2.33. They all not equal to 24, so the you must call Finish: give_up_and_restart.
'''
        self.input_description = f"The real task input is: {data_list}"

        self.tool_names = ["play_24"]
        self.functions = [
            {
                "name": "play_24",
                "description": "make your current conbine with the format \"x operation y = z\" like \"1+2=3\", then I will tell you whether you win. This is the ONLY way to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1",
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
                "name": "Finish",
                "description": "If you think you get the result which can answer the task, call this function to give the final answer. Or, if you think you can't handle the task from this status, call this function to restart. Remember: you should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be showed to user, so final answer should contain enough information.",
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
        ]


        self.data_list = data_list
        self.restart()


    def restart(self):
        self.now_datas = self.data_list.copy()
        self.status = 0

    def check_success(self):
        return self.status
    
    def get_score(self):
        # print(node.io_state.to_json())
        success = do_24(self.now_datas)
        if success:
            success = (1+4-len(self.now_datas))*10
        else:
            success = len(self.now_datas)
        return success

    def to_json(self):
        js_obj = {
            "now_left_numbers": self.now_datas,
            "status": self.status,
            "score": self.get_score(),
        }
        return js_obj

    def step(self,action_name, action_input):
        if action_name == "Finish":
            # tryfinishaction inputjson}
            try:
                json_data = json.loads(action_input,strict=False)
            except: # 
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
                
                self.success = 1 # final_answer
                return "{\"response\":\"successfully giving the final answer.\"}", 3
            else:
                "{error:\"\"return_type\" is not a valid choice\"}", 2
        try:
            js_data = json.loads(action_input)
            assert "input" in js_data.keys()
            action_input = js_data["input"]
        except Exception as e:
            pass
            
        former_data_str = f"{self.now_datas}"
        # re_pattern = r"combine (\d+)[,]\D*(\d+) by ((\d+)(\D+)(\d+))"
        re_pattern = r"((\-?\(?[\d\.]+\)?)([\+\-\*\/ ]+?)(\-?\(?[\d\.]+\)?))[^\+\-\*\/=]*=\D*(\-?[\d\.]+).*"
        match_result = re.match(re_pattern,action_input)
        if match_result != None:
            d1 = eval(match_result.group(2))
            d2 = eval(match_result.group(4))
            '''
            
            '''
            if d1 not in self.now_datas:
                return f"{d1} not in left numbers {former_data_str}", -2
            if d2 not in self.now_datas:
                return f"{d2} not in left numbers {former_data_str}", -2
            if d1 == d2 and self.now_datas:
                counter = list( filter(lambda x:x==d1,self.now_datas))
                if len(counter) < 2:
                    return f"there are only one \"{d1}\" number in left numbers {former_data_str}", -2
            # '''
            # 
            # '''
            # d3 = int(match_result.group(4))
            # d4 = int(match_result.group(6))
            # if d3 not in [d1,d2]:
            #     return f"{d3} not appear in {[d1,d2]}, left numbers {former_data_str}", 0
            # if d4 not in [d1,d2]:
            #     return f"{d4} not appear in {[d1,d2]}, left numbers {former_data_str}", 0

            try:
                result = eval(match_result.group(1))
                # if type(result) == float and not result.is_integer():
                #     return f"you combine numbers to {result}, which is not a integer, try other command. left numbers {former_data_str}", 0
                '''
                
                '''
                # result = int(result)
                ls1 = self.now_datas.index(d1)
                self.now_datas = self.now_datas[:ls1] + self.now_datas[ls1+1:]
                ls2 = self.now_datas.index(d2)
                self.now_datas = self.now_datas[:ls2] + self.now_datas[ls2+1:] 
                self.now_datas.append(result)  
                obs = f"{match_result.group(1)}={result}, left numbers: {self.now_datas}"

                
                if len(self.now_datas) == 1:
                    if self.now_datas[0] == 24:
                        self.status = 1
                        return f"{obs}, you win", 1
                    else:
                        self.status = -1
                        return f"{obs}, you lose", -1
                else:
                    return obs, 0
            except Exception as e:
                return str(e), -2
        else:
            return '''format error, please use format \"x operations y=result\"''', -2



standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''

def test_all(method):
    file_name = r"/mnt/d/git_repos/AutoToolLearning/LLM-AlphaGo/Downstream_tasks/24.csv"
    output_dir = rf"/mnt/d/git_repos/data/24/play24_{method}"
    os.makedirs(output_dir,exist_ok=True)
    data_list = []
    with open(file_name,"r") as f:
        for line in f.readlines()[1:]:
            temp_data = line.strip().split(",")[1].split(" ")
            temp_data = [int(cont) for cont in temp_data]
            data_list.append(temp_data)
    
    data_list = data_list[:100]
    total = 0
    win = 0
    for data in data_list:
        total += 1

        flatten_input = "_".join([str(cont) for cont in data])
        if os.path.exists(os.path.join(output_dir,f"{flatten_input}.json")):
            with open(os.path.join(output_dir,f"{flatten_input}.json"),"r",encoding="utf-8") as reader:
                json_obj = json.load(reader)
                if json_obj["win"] == True:
                    win += 1
            continue


        env, chain = test_24(data,method)
        if chain.status == 1:
            win += 1
            print(colored(f"You Win, ratio={win}/{total}={win/total}","green"))
        else:
            print(colored(f"You Lose, ratio={win}/{total}={win/total}","green"))

        with open(os.path.join(output_dir,f"{flatten_input}.json"),"w",encoding="utf-8") as writer:
            json_obj = chain.to_json(answer = False, process = True)
            json.dump(json_obj,writer,indent=2)


def check_success(data,line):
    # if not line.endswith("24"):
    #     return False
    
    re_pattern = "answer:([^=]+)=.*"
    re_result = re.match(re_pattern,line.lower())
    if re_result == None:
        return False
    formula = re_result.group(1)
    print(formula)
    re_pattern2 = "\D*(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D*"
    re_result2 = re.match(re_pattern2,formula)
    if re_result2 == None:
        return False
    numbers = [re_result2.group(1),re_result2.group(2),re_result2.group(3),re_result2.group(4)]
    numbers = [eval(d) for d in numbers]
    now_data = data.copy()
    for number in numbers:
        if number not in now_data:
            return False
        id = now_data.index(number)
        now_data = now_data[:id] + now_data[id+1:]
    output = eval(formula)
    if output != 24:
        return False
    return True

# suc = check_success([1,1,4,6],"Answer: (1 * 2) * (4 * 6) = 24")
# print(suc)

def check_all_io_prompt():
    file_name = r"C:\git_repos\LLM-AlphaGo\Downstream_tasks\24.csv"
    output_dir = rf"C:\git_repos\LLM-AlphaGo\test_result\play24_io_prompt_900_1000"
    os.makedirs(output_dir,exist_ok=True)
    data_list = []
    with open(file_name,"r") as f:
        for line in f.readlines()[1:]:
            temp_data = line.strip().split(",")[1].split(" ")
            temp_data = [int(cont) for cont in temp_data]
            data_list.append(temp_data)
    
    data_list = data_list[900:1000]
    total = 0
    win = 0
    for data in data_list:
        total += 1

        flatten_input = "_".join([str(cont) for cont in data])
        if os.path.exists(os.path.join(output_dir,f"{flatten_input}.json")):
            with open(os.path.join(output_dir,f"{flatten_input}.json"),"r",encoding="utf-8") as reader:
                json_obj = json.load(reader)
                if json_obj["win"] == True:
                    win += 1
            continue

        llm = chatGPT()
        prompt = standard_prompt.replace("{input}"," ".join([str(cont) for cont in data]))
        output = llm.generation("",prompt)
        print(colored(output,"yellow"))
        for line in output.split("\n"):
            suc = check_success(data,line)
            if suc:
                break
        if suc:
            win += 1
            print(colored(f"You Win, ratio={win}/{total}={win/total}","green"))
        else:
            print(colored(f"You Lose, ratio={win}/{total}={win/total}","green"))
        
        obj = {
            "input": data,
            "output": output,
            "win": suc,
        }
        with open(os.path.join(output_dir,f"{flatten_input}.json"),"w",encoding="utf-8") as writer:
            json.dump(obj,writer,indent=2)

def test_24(list_data,method):
    print(colored(f"now playing {list_data}","green"))
    env = play_24(list_data)

    # llm = chatGPT()
    # llm_forward = partial(llm.generation,model="text-davinci-003")
    llm_forward = chatgpt_0613()
    if method == "CoT":
        chain = single_chain(pass_at=1,llm=llm_forward,io_func=env)
        result = chain.start(single_chain_max_step=18)
    elif method == "Reflexion":
        chain = reflexion_chain( llm=llm_forward,io_func=env)
        result = chain.start(max_chain_count=5,single_chain_max_step=18)
    elif method == "BFS":
        chain = BFS_tree_search(llm=llm_forward,io_func=env)
        result = chain.start(search_width=3, expansion_ratio=3, single_chain_max_step=18,max_iters=4)
    elif method == "UCT_abstract":
        chain = UCT_abstract(llm=llm_forward,io_func=env)
        result = chain.start(simulation_count=10,single_chain_max_step=18)
    elif method == "UCT_vote":
        if callable(llm_forward):
            chain = UCT_vote(llm=llm_forward,io_func=env)
        else:
            chain = UCT_vote_function(llm=llm_forward,io_func=env)
        result = chain.start(simulation_count=10,epsilon_new_node=0.3,choice_count=3,vote_candidates=3,vote_count=1,single_chain_max_step=18)
    elif method == "ETS":
        chain = ETS_tree_search(llm=llm_forward, io_func=env,process_id=0)
        result = chain.start(simulation_count=5,
                            p_new_node=0.3,
                            matching_interval=1,
                            choice_count=5,
                            single_chain_max_step=18,
                            max_query_count = 25)
    return env, chain






if __name__ == "__main__":
    # env = play_24([6,7,8,9])
    # env.restart()
    # print(env.step(action_name="", action_input="8-9=-1"))
    # print(env.step(action_name="", action_input="7*-1=-1"))
    # exit()
    # re_pattern = r"\"candidate[ \_](\d+)\""
    # re_result = re.findall(re_pattern,"Best candidate: \"candidate 1\" as they were able to reach 24, while candidate 0 was unable to.")
    # print(re_result)
    # if re_result != []:
    #     assert len(re_result) == 1
    # exit()

    test_all("ETS")
    # check_all_io_prompt()

    # test_24_reflexion([1,2,4,7])
    # print(do_24([3, 7, 9, 13],[]))