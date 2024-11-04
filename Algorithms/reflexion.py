from Algorithms.single_chain import single_chain
import re
from Tree.Tree import my_tree, tree_node
import json
from Prompts.Reflexion_prompts import MAKE_REFLEXION_USER_PROMPT,CAT_REFLEXION_USER_PROMPT
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
from termcolor import colored
from Algorithms.base_search import base_search_method
from copy import deepcopy
import random

class reflexion_chain(base_search_method):

    def __init__(self,llm,io_func,process_id):
        super(reflexion_chain, self).__init__()
        '''
        sequence_lengthtryreflection

        '''
        self.io_func = io_func
        self.llm = llm

        self.single_chain_tree_list = []
        self.reflexion_list = []

        self.process_id = process_id

        self.terminal_node = []
        self.give_up_node = []
        
        self.restart()



    def to_json(self, answer=False,process=True):
        if process:
            json_obj = {
                "win": self.status == 1,
                "try_count": len(self.single_chain_tree_list),
                "trys": [tree.chain_tree_to_json() for tree in self.single_chain_tree_list],
                "reflections": self.reflexion_list,
                "compare_candidates": [],
                "forward_args":self.forward_args,
            }
            for node in self.terminal_node:
                if node.pruned == False: #
                    json_obj["compare_candidates"].append(node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "final_answer": "",
                "finish_type":"give_answer",
                "function": self.io_func.functions,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "train_messages": [],
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False: #
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["final_answer"] = node.description
                    # json_obj["answer_generation"]["chain"] = node.get_chain_result_from_this_node(use_messages=True)
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
            if json_obj["answer_generation"]["valid_data"] == False: #final answergive_up
                if len(self.give_up_node) > 0:
                    random_pos = random.randint(0,len(self.give_up_node) - 1)
                    choose_give_up_node = self.give_up_node[random_pos]
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_up"
                    json_obj["answer_generation"]["final_answer"] = choose_give_up_node.description
                    # json_obj["answer_generation"]["chain"] = node.get_chain_result_from_this_node(use_messages=False)
                    json_obj["answer_generation"]["train_messages"] = choose_give_up_node.get_train_messages_from_this_node()

        return json_obj

    def restart(self):

        self.tree = my_tree()
        self.status = 0
        self.query_count = 0
        self.total_tokens = 0

    def start(self,max_chain_count, single_chain_max_step, max_query_count):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")


        while True:
            '''
            single_chain
            reflection
            '''
            start_message_list = []
            system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
            system = system.replace("{task_description}",self.io_func.task_description)
            start_message_list.append({"role":"system","content":system})

            user = FORMAT_INSTRUCTIONS_USER_FUNCTION
            user = user.replace("{input_description}",self.io_func.input_description)
            start_message_list.append({"role":"user","content":user})

            reflection_prompts = ""
            for k, reflection in enumerate(self.reflexion_list):
                reflection_prompts += f"Reflection_for_try_{k+1}: {reflection}\n"
            if reflection_prompts == "":
                reflection_prompts = "None"
            user = CAT_REFLEXION_USER_PROMPT
            user = user.replace("{reflexion}",reflection_prompts)
            user = user.replace("{input_description}",self.io_func.input_description)
            start_message_list.append({"role":"user","content":user})

            # self.llm.change_messages(start_message_list)
            # self.llm.display_conversation()

            temp_single_chain = single_chain(start_message_list = start_message_list, llm=self.llm, io_func=self.io_func,process_id=self.process_id)
            
            temp_single_chain.start(single_chain_max_step = single_chain_max_step,pass_at=1)
            self.terminal_node.extend(temp_single_chain.terminal_node)
            self.give_up_node.extend(temp_single_chain.give_up_node)
            self.single_chain_tree_list.append(temp_single_chain.tree)
            self.query_count += temp_single_chain.query_count
            self.total_tokens += temp_single_chain.total_tokens

            self.make_reflexion(temp_single_chain.terminal_node[0]) 
            
            if temp_single_chain.status == 1: 
                '''
                single_chain
                '''
                self.status = 1
 
            if len(self.single_chain_tree_list) >= max_chain_count: #
                # exit()
                return 0  

            if self.query_count > max_query_count:
                return 0
            
            print(f"[Reflexion]restart the task for the {len(self.single_chain_tree_list)+1} time")
            self.io_func.restart()

            

    def make_reflexion(self,end_node):
        make_reflection_prompt = MAKE_REFLEXION_USER_PROMPT

        new_message = {
            "role": "user",
            "content":make_reflection_prompt,
        }

        message_list = deepcopy(end_node.messages)
        message_list.append(new_message)

        self.llm.change_messages(message_list)
        # self.llm.display_conversation()
        new_message,error_code,total_tokens = self.llm.parse(self.io_func.functions,function_call="none",process_id=self.process_id)
        self.query_count += 1
        self.total_tokens += total_tokens
        message_list.append(new_message) 
        if self.process_id == 0:
            print(colored(f"Reflexion: {new_message['content']}","red"))
        self.reflexion_list.append(new_message["content"].replace("\n\n","\n"))

        return 
    