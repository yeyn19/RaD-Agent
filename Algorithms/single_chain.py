'''


'''
import re
from Tree.Tree import my_tree, tree_node
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
from Algorithms.base_search import base_search_method
from copy import deepcopy
import random

class single_chain(base_search_method):

    def __init__(self,llm,io_func,process_id=0,start_message_list=None):
        super(single_chain, self).__init__()
        '''
         thought, action, action input
        '''
        self.io_func = io_func
        self.llm = llm
        self.start_message_list = start_message_list
        self.process_id = process_id

        self.restart()
    def restart(self):

        # self.tree = my_tree()
        self.status = 0
        self.try_list = []
        self.terminal_node = []
        self.give_up_node = []

        self.query_count = 0 #openai
        self.total_tokens = 0
        self.success_count = 0

    def to_json(self, answer=False,process=True):
        if process:
            json_obj = {
                "win": self.status == 1,
                "try_count": len(self.try_list),
                "trys": self.try_list,
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

    def to_json_single(self):
        json_obj = {}
        tree_obj = self.terminal_node[-1].get_chain_result_from_this_node()
        json_obj["chain"] = tree_obj
        json_obj["win"] = self.status == 1

        return json_obj



    def start(self,single_chain_max_step,max_query_count,pass_at=1,answer=1):
        self.forward_args = locals()
        assert max_query_count % 10 == 0
        self.json_list = []

        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")

        for i in range(pass_at):
            print(f"[process({self.process_id})][single_chain]try for the {i+1} time")
            self.tree = my_tree()
            self.tree.root.node_type = "Action Input"
            self.tree.root.io_state = deepcopy(self.io_func)
            out_node = self.do_chain(self.tree.root, single_chain_max_step)
            self.terminal_node.append(out_node)
            if out_node.observation_code == 4:
                self.give_up_node.append(out_node)
            self.try_list.append(self.to_json_single())
            if out_node.io_state.check_success() == 1:
                self.status = 1
                self.success_count += 1
                if self.success_count >= answer:
                    return 1
            if self.query_count > max_query_count:
                return 0
        return 0

    def do_chain(self,now_node,single_chain_max_step):
        if callable(self.llm):
            return self.do_chain_react(now_node,single_chain_max_step)
        else:
            return self.do_chain_function(now_node,single_chain_max_step)

    def do_chain_function(self,now_node,single_chain_max_step):
        '''
        0613
        '''

        '''
        rootself.messagessystemuser
        '''
        if self.start_message_list == None:
            system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
            system = system.replace("{task_description}",self.io_func.task_description)
            self.tree.root.messages.append({"role":"system","content":system})

            user = FORMAT_INSTRUCTIONS_USER_FUNCTION
            user = user.replace("{input_description}",self.io_func.input_description)
            self.tree.root.messages.append({"role":"user","content":user})
        else:
            self.tree.root.messages = self.start_message_list
            # print(self.tree.root.messages)
        
        now_node = self.tree.root
        while True:
            self.llm.change_messages(now_node.messages)
            new_message,error_code,total_tokens = self.llm.parse(functions=self.io_func.functions,process_id=self.process_id)
            self.total_tokens += total_tokens
            self.query_count += 1
            if self.query_count % 30 == 0:
                self.json_list.append(self.to_json(answer=True,process=True))
            assert new_message["role"] == "assistant"
            if "content" in new_message.keys() and new_message["content"] != None:
                # print(new_message["content"])
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = new_message["content"]
                child_io_state = deepcopy(now_node.io_state)
                
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if error_code != 0:
                    now_node.observation_code = error_code
                    now_node.pruned = True

            if "function_call" in new_message.keys():
                function_name = new_message["function_call"]["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(now_node.io_state)
                
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)

                temp_node.print(self.process_id)
                now_node = temp_node

                function_input = new_message["function_call"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(now_node.io_state)

                observation, status = child_io_state.step(action_name=now_node.description, action_input=function_input)
                now_node.action_history.append({"action_name": now_node.description, "action_input": function_input})
                
                temp_node.observation = observation
                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if status != 0: # 
                    # 0
                    # 1api
                    # 2
                    # 3final answer
                    # 4
                    if status == 4:
                        now_node.pruned = True
                    elif status == 1: #message
                        assert "function_call" in new_message.keys()
                        new_message["function_call"]["name"] = "invalid_hallucination_function_name"

                    # now_node.messages.append(new_message)
                    # return now_node
                
            
            now_node.messages.append(new_message)
            if now_node.node_type == "Action Input":
                now_node.messages.append({
                    "role":"function",
                    "name": new_message["function_call"]["name"],
                    "content": now_node.observation,
                })
            if now_node.get_depth() >= single_chain_max_step and not (now_node.is_terminal):
                # print(now_node.to_json())
                now_node.pruned = True
            
            if now_node.pruned or now_node.is_terminal:
                return now_node

    
