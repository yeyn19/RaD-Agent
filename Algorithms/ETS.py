from Tree.Tree import my_tree, tree_node
from copy import deepcopy
from Algorithms.base_search import base_search_method
from Prompts.Tree_search_prompts import  DIVERSITY_PROMPT,DEFAULT_POLICY_SYSTEM_PROMPT, DEFAULT_POLICY_USER_PROMPT

from LLM_rank.rank_candidate import elo_rank,sum_based_rankn,rank2_subfix,rank2_allchain

from termcolor import colored
import numpy as np
import re
import json
import math

from pprint import pprint
import pdb

from utils import softmax_bias, compute_epsilon_new_node
from functools import partial


class ETS_tree_search(base_search_method):
    def __init__(self,llm,io_func,process_id=0,):
        super(ETS_tree_search, self).__init__()
        '''
        Elo
        reflexion
        '''

        self.llm = llm
        self.io_func = io_func
        self.process_id = process_id
        self.restart()

    def to_json(self,answer=False, process=True):
        if process:
            js_obj = {
                "win": self.status == 1,
                "simulation_count": self.now_simulation_count,
                "forward_args":self.forward_args,
                "tree":self.tree.to_json_recursive(),
                "compare_candidates": [],
            }
            for node in self.terminal_node:
                if node.pruned == False: #
                    js_obj["compare_candidates"].append(node.get_chain_result_from_this_node(use_messages=False))
        else:
            js_obj = {}
        
        if answer:
            js_obj["answer_generation"] = {
                "valid_data": False,
                "final_answer": "",
                "finish_type":"give_answer",
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "forward_query_count": self.forward_query_count,
                "backward_query_count": self.backward_query_count,
                "function": self.io_func.functions,
            }
            if len(self.terminal_node) > 0: #value
                final_terminal_node = sorted(self.terminal_node, key=lambda x: x.Elo, reverse=True)[0]
                if final_terminal_node.pruned == False:
                    js_obj["answer_generation"]["valid_data"] = True
                    js_obj["answer_generation"]["final_answer"] = final_terminal_node.description
                    js_obj["answer_generation"]["train_messages"] = final_terminal_node.get_train_messages_from_this_node()
        
        return js_obj

    def restart(self): # tree
        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)

        prefix = DEFAULT_POLICY_SYSTEM_PROMPT
        prefix = prefix.replace("{task_description}",self.io_func.task_description)
        self.tree.root.messages.append({
            "role":"system",
            "content": prefix,
        })

        prefix = DEFAULT_POLICY_USER_PROMPT
        prefix = prefix.replace("{input_description}",self.io_func.input_description)
        self.tree.root.messages.append({
            "role":"user",
            "content": prefix,
        })


        self.status = 0
        self.now_simulation_count = 0
        self.terminal_node = []
        self.forward_query_count = 0
        self.backward_query_count = 0
        self.total_tokens = 0
        self.expand_num = 1

        self.json_list = []

    @property
    def query_count(self):
        return self.forward_query_count + self.backward_query_count
    
    def query_add(self, forward_add_num,backward_add_num):
        self.forward_query_count += forward_add_num
        self.backward_query_count += backward_add_num

        later_factor = self.query_count // 30
        while len(self.json_list) < later_factor:
            self.json_list.append(self.to_json(answer = True, process = True))

    def start(self,
              simulation_count, # chain
              temperature, #
              p_new_node, # 
              max_child_count, #
              filter_size, # best-of-N
              matching_interval, #Elo
              single_chain_max_step, #
              max_query_count, #
              Elo_args, #
            ):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")

        epsilon_new_node = compute_epsilon_new_node(p_new_node,temperature)

        self.max_query_count = max_query_count
        while self.now_simulation_count < simulation_count:

            print(f"[process({self.process_id})]simultation {self.now_simulation_count}")
            '''
            
            '''
            now_node = self.tree.root
            randomly_go_to_terminal_count = 0
            while len(now_node.children) > 0:
                '''
                
                '''
                # decision = self.make_decision(now_node)
                decision = self.make_decision_by_value(now_node, epsilon_new_node,max_child_count,temperature)
                if decision == "early_stopping":
                    return 0


                if decision == -1:
                    if self.process_id == 0:
                        print(colored("decide to make new node!","green"))
                    break
                if self.process_id == 0:
                    print(colored(f"decide to go down child {decision}","green"))

                now_node = now_node.children[decision]
                while now_node.node_type != "Action Input" and len(now_node.children) > 0:
                    now_node = now_node.children[0]

            if now_node.is_terminal or now_node.pruned:
                if self.process_id == 0:
                    print(colored(f"randomly go down to terminal nodes","green"))
                randomly_go_to_terminal_count += 1
                if randomly_go_to_terminal_count > 100: #100
                    return 0
            else:
                end_node = self.default_policy(now_node,single_chain_max_step,filter_size)
                if type(end_node) == str and end_node == "early_stopping":
                    return 0

                self.now_simulation_count += 1

                end_node.init_Elo()

                self.terminal_node.append(end_node)

                if end_node.io_state.check_success() == 1:
                    self.status = 1
                    # self.llm.display_conversation()
                    # return 1


                '''
                candidate
                '''
                if self.now_simulation_count % matching_interval == 0 and len(self.terminal_node) >= 2:
                    LLM_rank_args = {
                        "functions": self.io_func.functions,
                        "process_id": self.process_id,
                        "task_description": self.io_func.task_description,
                        "input_description": self.io_func.input_description,
                        "rank_func": rank2_allchain,
                        
                        
                    }
                    new_candidate_pos = list(range(len(self.terminal_node))[-matching_interval:])
                    balence_func = partial(self.tree.balence_Elo,temperature=temperature)
                    output,Elo_query_count,total_tokens = elo_rank(self.llm,LLM_rank_args, self.terminal_node,new_candidate_pos,balence_func=balence_func,Elo_args=Elo_args,root_node=self.tree.root)

                    self.total_tokens += total_tokens
                    self.query_add(0,Elo_query_count)

                    if self.query_count > max_query_count:
                        return 0

        return 0




    def make_decision_by_value(self, now_node, epsilon_new_node, max_child_count, temperature):
        '''
        ELo
        -1
        finish
        filter""
        '''


        elos = [-10000 if (child.expand_num == 0 or child.finished) else child.Elo for child in now_node.children] 
        if len(now_node.children) < max_child_count:
            elos.append(epsilon_new_node)
        temperature  = now_node.compute_choice_temperature(temperature)
        weights = softmax_bias(elos,temperature)
        if self.process_id == 0:
            print(f"Elo: ",end="")
            for elo in elos:
                print(f"{elo:.2f}",end=" ")
            print()
            print(f"Weights(e={now_node.matching_time}, t={temperature}): ",end="")
            for weight in weights:
                print(f"{weight:.2f}",end=" ")
            print()
        result = np.random.multinomial(1,weights)
        for k, v in enumerate(result[:-1]):
            if v == 1:
                return k
        return -1


    def default_policy(self,now_node,single_chain_max_step,filter_size):
        '''
        0613
        filter_size
        '''
        assert (not now_node.is_terminal) and (not now_node.pruned)
        assert now_node.messages != []


        while True:
            
            '''
            
            '''
            if len(now_node.children) > 0:
                for k, child in enumerate(now_node.children):
                    if child.expand_num == 0: #
                        if self.process_id == 0:
                            print(f"use former_generated false_filterd path, id={k}")
                        now_node = now_node.children[k]
                        now_node.expand_num = self.expand_num
                        self.expand_num += 1
                        while len(now_node.children) > 0:
                            assert len(now_node.children) == 1, f"len = {len(now_node.children)}"
                            now_node = now_node.children[0]
                            now_node.expand_num = self.expand_num
                            self.expand_num += 1
                        if now_node.get_depth() >= single_chain_max_step and not (now_node.is_terminal):
                            now_node.pruned = True #

                        if now_node.is_terminal or now_node.pruned:
                            return now_node

            '''
            
            filter filter*N
            '''
            new_generated_list = []
            for _ in range(filter_size):
                '''
                DIVERSITY_PROMPTmessage
                '''
                temp_now_node = now_node
                if self.process_id == 0:
                    print(f"generating for depth {temp_now_node.get_depth()}, child {len(temp_now_node.children)}")

                use_diversity_prompt = False
                if len(temp_now_node.children) > 0:
                    diverse_prompt = DIVERSITY_PROMPT
                    former_candidates_des = ""
                    js_list = []
                    for k, child in enumerate(temp_now_node.children):
                        temp_node = child
                        while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                            temp_node = temp_node.children[0]
                        # child_des = self.get_former_trice(child,temp_node)
                        # former_candidates_des = former_candidates_des + f"<candidate_{k+1}>\n{child_des}"
                        if temp_node.node_type == "Action Input":
                            try:
                                arguments = json.loads(temp_node.description)
                            except:
                                arguments = temp_node.description
                            obj_dict = {
                                "name": temp_node.father.description,
                                "arguments": arguments,
                                "function_output": temp_node.observation,
                                "mento-carlo-action-value": temp_node.compute_weight(),
                            }
                            js_list.append(obj_dict)
                    
                    if len(js_list) > 0:
                        former_candidates_des = former_candidates_des + f"{json.dumps(js_list,indent=2)}\n"
                        if temp_now_node.observation != "":
                            former_candidates_des = former_candidates_des + f"again, your former observation: {temp_now_node.observation}\n"
                        diverse_prompt = diverse_prompt.replace("{previous_candidate}",former_candidates_des)
                        use_diversity_prompt = True
                        temp_now_node.messages.append({"role":"user", "content":diverse_prompt})
                        self.llm.change_messages(temp_now_node.messages)
                        # self.llm.display_conversation()

            
                self.llm.change_messages(temp_now_node.messages)
                # self.llm.display_conversation()
                new_message, error_code,total_tokens = self.llm.parse(self.io_func.functions,process_id=self.process_id)
                self.total_tokens += total_tokens
                self.query_add(1,0)
                if self.query_count >= self.max_query_count:
                    return "early_stopping"

                assert new_message["role"] == "assistant"

                '''
                diversity prompt
                '''
                if use_diversity_prompt:
                    temp_now_node.messages = temp_now_node.messages[:-1]

                if "content" in new_message.keys() and new_message["content"] != None:
                    # print(new_message["content"])
                    temp_node = tree_node()
                    temp_node.node_type = "Thought"
                    temp_node.description = new_message["content"]
                    child_io_state = deepcopy(temp_now_node.io_state)
                    
                    temp_node.io_state = child_io_state
                    temp_node.is_terminal = child_io_state.check_success() != 0 
                    temp_node.messages = deepcopy(temp_now_node.messages)
                    temp_node.father = temp_now_node
                    temp_now_node.children.append(temp_node)
                    temp_node.print(self.process_id)
                    temp_now_node = temp_node

                    if error_code != 0:
                        temp_now_node.observation_code = error_code
                        temp_now_node.pruned = True

                if "function_call" in new_message.keys():
                    function_name = new_message["function_call"]["name"]
                    # assert function_name in now_node.io_state.tool_names

                    # new the Action node
                    temp_node = tree_node()
                    temp_node.node_type = "Action"
                    temp_node.description = function_name
                    child_io_state = deepcopy(temp_now_node.io_state)
                    
                    temp_node.io_state = child_io_state
                    temp_node.is_terminal = child_io_state.check_success() != 0 
                    temp_node.messages = deepcopy(temp_now_node.messages)
                    temp_node.father = temp_now_node
                    temp_now_node.children.append(temp_node)

                    temp_node.print(self.process_id)
                    temp_now_node = temp_node


                    # new the Action Input and Observation node
                    function_input = new_message["function_call"]["arguments"]
                    temp_node = tree_node()
                    temp_node.node_type = "Action Input"
                    temp_node.description = function_input
                    child_io_state = deepcopy(temp_now_node.io_state)

                    observation, status = child_io_state.step(action_name=temp_now_node.description, action_input=function_input)
                    temp_node.observation = observation
                    temp_node.observation_code = status
                    temp_node.io_state = child_io_state
                    temp_node.is_terminal = child_io_state.check_success() != 0 
                    temp_node.messages = deepcopy(temp_now_node.messages)

                    temp_node.father = temp_now_node
                    temp_now_node.children.append(temp_node)
                    temp_node.print(self.process_id)
                    temp_now_node = temp_node

                    if status != 0: # 
                        # 0
                        # 1api
                        # 2
                        # 3final answer
                        # 4
                        # ...
                        if status == 4:
                            temp_now_node.make_finish(2)
                            temp_now_node.pruned = True
                        elif status == 3: #final_answer
                            temp_now_node.make_finish(2)
                            temp_now_node.is_terminal = True
                        elif status == 1: #message
                            assert "function_call" in new_message.keys()
                            new_message["function_call"]["name"] = "invalid_hallucination_function_name"    
                
                        

                temp_now_node.messages.append(new_message)
                if temp_now_node.node_type == "Action Input":
                    temp_now_node.messages.append({
                        "role":"function",
                        "name": new_message["function_call"]["name"],
                        "content": temp_now_node.observation,
                    })
                new_generated_list.append(temp_now_node)

            '''
            new_generated_list
            '''
            assert len(new_generated_list) > 0
            if len(new_generated_list) > 1:
                '''
                next_tree_split_nodes
                '''
                LLM_rank_args = {
                    "functions": self.io_func.functions,
                    "process_id": self.process_id,
                    "task_description": self.io_func.task_description,
                    "input_description": self.io_func.input_description,
                    "rank_func": rank2_subfix,
                }
                scores, rank_query_count,total_tokens,rank_details = sum_based_rankn(self.llm,LLM_rank_args=LLM_rank_args,candidates=new_generated_list)
                self.query_add(rank_query_count,0)
                self.total_tokens += total_tokens
                for score, node in zip(scores, new_generated_list):
                    node.prior_score = score
                zip_value = list(zip(new_generated_list,range(len(new_generated_list))))
                zip_value.sort(key=lambda x: x[0].prior_score, reverse=True) #score
                new_generated_list,filtered_order = zip(*zip_value)
                if self.process_id == 0:
                    print(f"scores={scores}, filtered order: {filtered_order}")

                select_child = new_generated_list[filtered_order[0]]
            else:
                select_child = new_generated_list[0]


            '''
            
            '''
            def reversed_get_expand_num(temp_node,end_node):
                if temp_node == end_node:
                    # temp_node.expand_num = self.expand_num
                    return self.expand_num
                father_expand_num = reversed_get_expand_num(temp_node.father,end_node)
                self.expand_num += 1
                temp_node.expand_num = self.expand_num
                return temp_node.expand_num
            reversed_get_expand_num(select_child,now_node)

            now_node = select_child


            if now_node.get_depth() >= single_chain_max_step and (not (now_node.is_terminal)):
                now_node.pruned = True #

            if now_node.is_terminal or now_node.pruned:
                return now_node

