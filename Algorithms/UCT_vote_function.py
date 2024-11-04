from Tree.Tree import my_tree, tree_node
from copy import deepcopy
from Algorithms.base_search import base_search_method
from Prompts.Tree_search_prompts import  DIVERSITY_PROMPT,VOTE_BEST_SYSTEM_PROMPT,VOTE_BEST_USER_PROMPT,DEFAULT_POLICY_SYSTEM_PROMPT, DEFAULT_POLICY_USER_PROMPT,DEFAULT_POLICY_STATE_CHANGE_USER_PROMPT
from Prompts.Reflexion_prompts import MAKE_REFLEXION_USER_PROMPT
from termcolor import colored
import numpy as np
import re
import json

from pprint import pprint
import pdb

class UCT_vote_function(base_search_method):
    def __init__(self,llm,io_func,process_id=0):
        super(UCT_vote_function, self).__init__()
        '''
        :
        1.0613
        2.value
        2. 
        '''

        self.llm = llm
        self.io_func = io_func
        self.simulations = []
        self.process_id = process_id
        self.restart()

    def to_json(self,answer_only=False):
        if not answer_only:
            js_obj = {
                "win": self.status == 1,
                "simulation_count": self.now_simulation_count,
                "simulations": self.simulations,
                "forward_args":self.forward_args,
                "tree":self.tree.to_json_recursive()
            }
        else:
            js_obj = {}

        js_obj["answer_generation"] = {
            "valid_data": False,
            "final_answer": "",
            "query_count": self.query_count,
            "function": self.io_func.functions,
            "chain": [],
        }


        if len(self.terminal_node) > 0: #value
            final_terminal_node = sorted(self.terminal_node, key=lambda x: sum(x.values)/(len(x.values)+1e-8), reverse=True)[0]
            js_obj["answer_generation"]["valid_data"] = True
            js_obj["answer_generation"]["final_answer"] = final_terminal_node.description
            # js_obj["answer_generation"]["chain"] = final_terminal_node.get_chain_result_from_this_node()
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
        self.simulations = []
        self.terminal_node = []
        self.total_vote = 0
        self.good_vote = 0 
        self.query_count = 0
        self.expand_num = 0

    def start(self,
              simulation_count, # chain
              epsilon_new_node, # value
              choice_count, # vote
              vote_candidates, # vote
              vote_count, # 
              single_chain_max_step, #
              max_query_count, #
            ):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")

        self.max_query_count = max_query_count
        while self.now_simulation_count < simulation_count:
            # print(colored("new trail start!","yellow"))
            '''
            
            '''
            now_node = self.tree.root
            while len(now_node.children) > 0:
                '''
                
                '''
                # decision = self.make_decision(now_node)
                decision = self.make_decision_by_value(now_node, epsilon_new_node)
                if decision == "early_stopping":
                    return 0


                if decision == -1:
                    print(colored("decide to make new node!","green"))
                    break
                print(colored(f"decide to go down child {decision}","green"))

                now_node = now_node.children[decision]
                while now_node.node_type != "Action Input" and len(now_node.children) > 0:
                    now_node = now_node.children[0]

            if now_node.is_terminal:
                print(colored(f"randomly go down to terminal nodes","green"))
            else:
                begin_default_policy_node = now_node
                end_node = self.default_policy(now_node,single_chain_max_step)
                if type(end_node) == str and end_node == "early_stopping":
                    return 0

                self.now_simulation_count += 1
                if end_node.pruned is not True: #
                    self.terminal_node.append(end_node)

                if end_node.io_state.check_success() == 1:
                    self.status = 1
                    # self.llm.display_conversation()
                    return 1

                '''
                
                '''
                output = self.make_reflection(begin_default_policy_node,end_node)
                if output == "early_stopping":
                    return 0

            '''
            candidate
            '''
            output = self.vote(choice_count,vote_candidates,vote_count)
            if output == "early_stopping":
                return 0

        return 0

    def vote(self,choice_count,vote_candidates,vote_count):
        '''
        
        vote_candidates
        vote_count
        '''
        if len(self.terminal_node) < vote_candidates:
            return
        

        for choice_count in range(choice_count):
            ordered = list(range(len(self.terminal_node)))
            np.random.shuffle(ordered)

            choices = ordered[:vote_candidates] #
            choices.sort()
            messages = []
            prompt = VOTE_BEST_SYSTEM_PROMPT
            prompt = prompt.replace("{task_description}",self.io_func.task_description)
            messages.append({
                "role":"system",
                "content": prompt,
            })

            prompt = VOTE_BEST_USER_PROMPT
            prompt = prompt.replace("{input_description}",self.io_func.input_description)


            candidates_description = ""
            for k, child_id in enumerate(choices):
                trice = self.tree.get_former_trice(self.tree.root,self.terminal_node[child_id],valid_types=["Action","Action Input","Observation"])
                candidates_description += f"<candidate_{k}>\n{trice}"
                # reflection = f"Reflection: {self.terminal_node[child_id].generated_reflection.strip()}\n"
                # candidates_description += reflection
                candidates_description += "*"*30 + "\n"
            prompt = prompt.replace("{candidate_description}",candidates_description)

            messages.append({
                "role":"user",
                "content": prompt,
            })

            # print(prompt)

            real_score = [-1]*len(choices)
            max_score = self.tree.root.io_state.get_score()
            max_position = -1
            for k, child_id in enumerate(choices):
                now_node = self.terminal_node[child_id]
                while now_node != None:
                    real_score[k] = max(now_node.io_state.get_score(),real_score[k])
                    now_node = now_node.father
                if real_score[k] > max_score:
                    max_position = k
                    max_score = real_score[k]
            # print(real_score)


            votes = [0]*len(choices)
            vaild_votes = 0
            for i in range(vote_count):
                '''
                
                '''
                self.llm.change_messages(messages)
                message, error_code = self.llm.parse(self.io_func.functions,function_call="none",process_id=self.process_id)
                self.query_count += 1
                if self.query_count >= self.max_query_count:
                    return "early_stopping"
                vote = message["content"]
                # print(vote)
                best_candiate_line = vote.split("\n")[-1]
                print(best_candiate_line)
                re_pattern = r"\"?candidate[ \_](\d+)\"?"
                re_result = re.findall(re_pattern,best_candiate_line.lower())
                if re_result != []:
                    if not len(re_result) == 1:
                        # print(best_candiate_line)
                        # exit()
                        return
                    vote_to = int(re_result[0])
                    self.total_vote += 1
                    if vote_to >= 0 and vote_to < len(votes):
                        votes[vote_to] += 1
                        self.good_vote += (vote_to == max_position)
                        vaild_votes += 1
                        print(colored(f"valid vote to {choices[vote_to]}","yellow"))
                    else:
                        self.good_vote += (-1 == max_position)
                        print(colored(f"vote to Invalid candidates, both candidate punished","yellow"))
                else:
                    self.good_vote += (-1 == max_position)
                    print(colored(f"vote to Nothing, both candidate punished","yellow"))
            if vaild_votes > 0:
                for k,child_id in enumerate(choices):
                    vote_count_this_turn = votes[k]
                    value = (vote_count_this_turn / vaild_votes  - 1 / vote_candidates) / np.sqrt(vaild_votes)
                    print(value)
                    now_node = self.terminal_node[child_id]
                    while now_node != None:
                        now_node.values.append(value)
                        now_node.vote_counts.append(vote_count_this_turn)
                        now_node = now_node.father  
        
        if self.total_vote > 0:
            print(f"ratio={self.good_vote}/{self.total_vote}={self.good_vote/self.total_vote}")

    def make_decision_by_value(self, now_node, epsilon_new_node):
        '''
        epsilon_new_node
        value
        '''
        # assert len(now_node.children) > 0
        # if np.random.random() < epsilon_new_node / len(now_node.children):
        #     return -1


        weights = [child.compute_weight() for child in now_node.children] + [epsilon_new_node]
        def my_softmax(x):
            exp_x = np.exp(x)
            return exp_x/np.sum(exp_x)
        
        weights = my_softmax(np.array(weights))
        print(weights)
        result = np.random.multinomial(1,weights)
        for k, v in enumerate(result[:-1]):
            if v == 1:
                return k
        return -1

    def make_reflection(self,start_node,end_node):
        '''
        start_nodeend_node,start_node
        '''
        make_reflection_prompt = MAKE_REFLEXION_USER_PROMPT

        new_message = {
            "role": "user",
            "content":make_reflection_prompt,
        }

        message_list = deepcopy(end_node.messages)
        message_list.append(new_message)

        self.llm.change_messages(message_list)
        new_message, error_code = self.llm.parse(self.io_func.functions,function_call="none",process_id=self.process_id)
        self.query_count += 1
        if self.query_count >= self.max_query_count:
            return "early_stopping"
        reflection = new_message['content'].replace("\n","")
        if self.process_id == 0:
            print(colored(f"Reflexion: {reflection}","green"))

        start_node.reflection.append(reflection)
        if start_node != self.tree.root:
            self.tree.root.reflection.append(reflection)


        return reflection

    def default_policy(self,now_node,single_chain_max_step):
        '''
        0613
        '''
        assert not now_node.is_terminal
        assert now_node.messages != []

        '''
         reflections
        '''
        reflections = ""
        if len(now_node.reflection) > 0:
            for k, ref in enumerate(now_node.reflection):
                reflections += f"Reflection_{k+1}: {ref}\n"
        else:
            reflections = "None\n"
        state_change_prompt = DEFAULT_POLICY_STATE_CHANGE_USER_PROMPT
        state_change_prompt = state_change_prompt.replace("{previous_reflections}", reflections)
        now_node.messages.append({
            "role": "user",
            "content": state_change_prompt,
        })


        first_time = True
        while True:

            if first_time:
                '''
                diversity prompt
                '''
                if len(now_node.children) > 0:
                    diverse_prompt = DIVERSITY_PROMPT
                    former_candidates_des = ""
                    js_list = []
                    for k, child in enumerate(now_node.children):
                        temp_node = child
                        while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                            temp_node = temp_node.children[0]
                        # child_des = self.get_former_trice(child,temp_node)
                        # former_candidates_des = former_candidates_des + f"<candidate_{k+1}>\n{child_des}"
                        if temp_node.node_type == "Action Input":
                            obj_dict = {
                                "name": temp_node.father.description,
                                "arguments": json.loads(temp_node.description),
                                "function_output": temp_node.observation,
                                "mento-carlo-action-value": temp_node.compute_weight(),
                            }
                            js_list.append(obj_dict)
                    
                    if len(js_list) > 0:
                        former_candidates_des = former_candidates_des + f"{json.dumps(js_list,indent=2)}\n"
                        if now_node.observation != "":
                            former_candidates_des = former_candidates_des + f"again, your former observation: {now_node.observation}\n"
                        diverse_prompt = diverse_prompt.replace("{previous_candidate}",former_candidates_des)
                        now_node.messages.append({"role":"user", "content":diverse_prompt})

                        self.llm.change_messages(now_node.messages)
                        # self.llm.display_conversation()
            
            self.llm.change_messages(now_node.messages)
            # self.llm.display_conversation()
            new_message, error_code = self.llm.parse(self.io_func.functions,process_id=self.process_id)
            self.query_count += 1
            if self.query_count >= self.max_query_count:
                return "early_stopping"

            assert new_message["role"] == "assistant"

            if first_time:
                first_time = False
                '''
                diversity prompt
                '''
                try:
                    if now_node.messages[-1]["content"][:20] == DIVERSITY_PROMPT[:20]:
                        now_node.messages = now_node.messages[:-1]
                except BaseException as e:
                    print(e)
                    pdb.set_trace()
                    pass

            if "content" in new_message.keys() and new_message["content"] != None:
                # print(new_message["content"])
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = new_message["content"]
                child_io_state = deepcopy(now_node.io_state)
                
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = deepcopy(now_node.messages)
                temp_node.father = now_node
                self.expand_num += 1
                temp_node.expand_num = self.expand_num
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if error_code != 0:
                    now_node.observation_code = error_code
                    now_node.pruned = True

            if "function_call" in new_message.keys():
                function_name = new_message["function_call"]["name"]
                # assert function_name in now_node.io_state.tool_names

                # new the Action node
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(now_node.io_state)
                
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = deepcopy(now_node.messages)
                self.expand_num += 1
                temp_node.expand_num = self.expand_num
                temp_node.father = now_node
                now_node.children.append(temp_node)

                temp_node.print(self.process_id)
                now_node = temp_node


                # new the Action Input and Observation node
                function_input = new_message["function_call"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(now_node.io_state)

                observation, status = child_io_state.step(action_name=now_node.description, action_input=function_input)
                temp_node.observation = observation
                temp_node.observation_code = status
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = deepcopy(now_node.messages)
                self.expand_num += 1
                temp_node.expand_num = self.expand_num
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

            
                    

            now_node.messages.append(new_message)
            if now_node.node_type == "Action Input":
                now_node.messages.append({
                    "role":"function",
                    "name": new_message["function_call"]["name"],
                    "content": now_node.observation,
                })

            if now_node.get_depth() >= single_chain_max_step:
                now_node.pruned = True #

            if now_node.is_terminal or now_node.pruned:
                
                '''
                state_change_prompt
                '''
                temp_node = now_node
                while temp_node != None:
                    for k in range(len(temp_node.messages)):
                        if type(temp_node.messages[k]["content"]) == str and temp_node.messages[k]["content"][:20] == DEFAULT_POLICY_STATE_CHANGE_USER_PROMPT[:20]:
                            temp_node.messages = temp_node.messages[:k] + temp_node.messages[k+1:]
                            break
                    temp_node = temp_node.father
                return now_node

