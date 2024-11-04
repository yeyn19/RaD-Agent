
from Tree.Tree import my_tree, tree_node
from copy import deepcopy
from Algorithms.base_search import base_search_method
from Prompts.Tree_search_prompts import  DIVERSITY_PROMPT
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
import json
import random

from LLM_rank.rank_candidate import get_best_N
from LLM_rank.rank_candidate import rank2_allchain

class BFS_tree_search(base_search_method):
    
    def __init__(self,llm,io_func,process_id):
        super(BFS_tree_search, self).__init__()
        '''
        
        topk
        '''

        self.llm = llm
        self.process_id = process_id

        self.io_func = io_func

        self.restart()
    
    def query_add(self, forward_add_num,backward_add_num):
        self.forward_query_count += forward_add_num
        self.rank_query_count += backward_add_num

        later_factor = (self.forward_query_count + self.rank_query_count) // 30
        while len(self.json_list) < later_factor:
            self.json_list.append(self.to_json(answer = True, process = True))


    @property
    def query_count(self):
        return self.forward_query_count + self.rank_query_count

    def restart(self):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")
        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)
        
        '''
        rootself.messagessystemuser
        '''
        system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        system = system.replace("{task_description}",self.io_func.task_description)
        self.tree.root.messages.append({"role":"system","content":system})

        user = FORMAT_INSTRUCTIONS_USER_FUNCTION
        user = user.replace("{input_description}",self.io_func.input_description)
        self.tree.root.messages.append({"role":"user","content":user})


        self.pool = [self.tree.root]
        self.terminal_node = []
        self.give_up_node = []

        self.forward_query_count = 0
        self.rank_query_count = 0
        self.total_tokens = 0

        self.status = 0
        self.expand_num = 1

        self.json_list = []


    def to_json(self,answer=False,process=True):

        if process:
            json_obj = {
                "win": self.status == 1,
                "tree": self.tree.to_json_recursive(),
                "forward_args":self.forward_args,
                "compare_candidates": [],
            }
            for node in self.terminal_node:
                if node.pruned == False: #
                    json_obj["compare_candidates"].append(node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

            
        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "query_count": self.query_count,
                "forward_query_count": self.forward_query_count,
                "rank_query_count": self.rank_query_count,
                "total_tokens": self.total_tokens,
                "final_answer": "",
                "finish_type":"give_answer",
                "function": self.io_func.functions,
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False: #
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_answer"
                    json_obj["answer_generation"]["final_answer"] = node.description
                    # json_obj["answer_generation"]["chain"] = node.get_chain_result_from_this_node(use_messages=False)
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

    def start(self,tree_beam_size,expansion_ratio,single_chain_max_step, max_query_count, answer=1):
        '''
        tree_beam_sizetopk
        expansion_ratio
        single_chain_max_step
        max_query_countquery
        answer

        
        '''

        while True:
            '''
            sample k
            '''
            new_pool = []
            for node in self.pool:
                new_N_node = self.make_candidates(node,expansion_ratio,max_query_count)
                if type(new_N_node) == str and new_N_node == "early_stopping":
                    return 0
                new_pool.extend(new_N_node)

            for node in new_pool:
                '''
                
                '''
                if node.is_terminal:
                    self.terminal_node.append(node)
                elif node.observation_code == 4:
                    self.give_up_node.append(node)
            if len(self.terminal_node) >= answer:
                return 0


            '''
            
            '''
            sifted_new_pool = []
            for k, node in enumerate(new_pool):
                if not (node.is_terminal or node.pruned or node.get_depth() >= single_chain_max_step):
                    sifted_new_pool.append(node)
        


            if len(sifted_new_pool) == 0:
                '''
                
                '''
                return 0
            elif len(sifted_new_pool) > tree_beam_size:
                '''
                k
                '''
                LLM_rank_args = {
                    "functions": self.io_func.functions,
                    "process_id": self.process_id,
                    "task_description": self.io_func.task_description,
                    "input_description": self.io_func.input_description,
                    "rank_func": rank2_allchain,
                }
                best_N_ids,rank_query_count,rank_total_tokens = get_best_N(self.llm,LLM_rank_args,sifted_new_pool,tree_beam_size)
                if self.process_id == 0:
                    print(f"sift {len(sifted_new_pool)}->{tree_beam_size} cost query_count {rank_query_count}")
                self.query_add(0,rank_query_count)
                self.total_tokens += rank_total_tokens
                if self.query_count > max_query_count:
                    return 0
                self.pool = []
                for left_id in best_N_ids:
                    self.pool.append(sifted_new_pool[left_id])
            else:
                self.pool = sifted_new_pool


    def make_candidates(self,node,expansion_ratio,max_query_count):
        '''
        nodeexpansion_ratio
        '''
        node.expand_num = self.expand_num
        self.expand_num += 1

        children_list = []
        for i in range(expansion_ratio):
            if self.process_id == 0:
                print(f"generating child {i} for depth {node.get_depth()}")
            temp_now_node = node
            # print(f"generate depth {temp_now_node.get_depth()} child {i}")
            '''
             diversity prompt
            '''
            delete_former_diversity_message = False
            diversity_message = None
            if len(temp_now_node.children) > 0:
                
                former_candidates_des = ""
                js_list = []
                for k, child in enumerate(temp_now_node.children):
                    temp_node = child
                    while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                        temp_node = temp_node.children[0]
                    # child_des = self.get_former_trice(child,temp_node)
                    # former_candidates_des = former_candidates_des + f"<candidate_{k+1}>\n{child_des}"
                    if temp_node.node_type == "Action Input":
                        obj_dict = {
                            "name": temp_node.father.description,
                            "arguments": temp_node.description,
                            "function_output": temp_node.observation,
                            "mento-carlo-action-value": temp_node.compute_weight(),
                        }
                        js_list.append(obj_dict)
                
                if len(js_list) > 0:
                    former_candidates_des = former_candidates_des + f"{json.dumps(js_list,indent=2)}\n"
                    if temp_now_node.observation != "":
                        former_candidates_des = former_candidates_des + f"again, your former observation: {temp_now_node.observation}\n"
                    diverse_prompt = DIVERSITY_PROMPT
                    diverse_prompt = diverse_prompt.replace("{previous_candidate}",former_candidates_des)
                    diversity_message = {"role":"user", "content":diverse_prompt}
                    temp_now_node.messages.append(diversity_message)

                    delete_former_diversity_message = True
                    # self.llm.change_messages(now_node.messages)
                    # self.llm.display_conversation()

            self.llm.change_messages(temp_now_node.messages)
            new_message,error_code,total_tokens = self.llm.parse(self.io_func.functions, process_id=self.process_id)
            self.query_add(1,0)
            self.total_tokens += total_tokens
            if (self.forward_query_count + self.rank_query_count) >= max_query_count:
                return "early_stopping"

            if delete_former_diversity_message:
                # temp_now_node.messages = temp_now_node.messages[:-1]
                temp_now_node.messages[-1]["valid"] = False

            assert new_message["role"] == "assistant"
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
                    if status == 4:
                        temp_now_node.pruned = True
                    elif status == 1: #message
                        assert "function_call" in new_message.keys()
                        new_message["function_call"]["name"] = "invalid_hallucination_function_name"
                    elif status == 3: # final answer
                        temp_now_node.is_terminal = True
                        self.status = 1

            temp_now_node.messages.append(new_message)
            if temp_now_node.node_type == "Action Input":
                temp_now_node.messages.append({
                    "role":"function",
                    "name": new_message["function_call"]["name"],
                    "content": temp_now_node.observation,
                })

            children_list.append(temp_now_node)
        return children_list
        