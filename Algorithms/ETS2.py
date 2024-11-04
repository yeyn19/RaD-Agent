from Tree.Tree import my_tree, tree_node
from copy import deepcopy
from Algorithms.base_search import base_search_method
from Prompts.Tree_search_prompts import  DIVERSITY_PROMPT,DEFAULT_POLICY_SYSTEM_PROMPT, DEFAULT_POLICY_USER_PROMPT

from LLM_rank.rank_candidate2 import elo_rank,sum_based_rankn,rank2_subfix,rank2_allchain, rank2_rule, rank2_oracle_rule

from termcolor import colored
import numpy as np
import re
import json
import math
from ets_utils import do_24

from pprint import pprint
import pdb

from ets_utils import softmax_bias, compute_epsilon_new_node
from functools import partial


class ETS_tree_search(base_search_method):
    def __init__(self,llm,io_func,process_id=0,):
        super(ETS_tree_search, self).__init__()
        '''
        仅由Elo积分驱动的树搜索
        不引入任何和reflexion相关的东西
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
                if node.pruned == False: #有答案
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
            if len(self.terminal_node) > 0: #选择value最高的
                final_terminal_node = sorted(self.terminal_node, key=lambda x: x.Elo, reverse=True)[0]
                if final_terminal_node.pruned == False:
                    js_obj["answer_generation"]["valid_data"] = True
                    js_obj["answer_generation"]["final_answer"] = final_terminal_node.description
                    js_obj["answer_generation"]["train_messages"] = final_terminal_node.get_train_messages_from_this_node()
        
        return js_obj

    def restart(self): # 理论上用不到，清空所有的tree
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
              simulation_count, # chain的个数
              temperature, #温度，增加随机性
              p_new_node, # 开新节点的初始概率
              max_child_count, #最大孩子节点数
              filter_size, # 先验的best-of-N
              matching_interval, #新出多少个叶子以后进行一轮Elo匹配
              single_chain_max_step, #单条链最大长度
              max_query_count, #最大访问次数
              Elo_args, #做匹配用到的参数
              subfix="",
            ):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")

        epsilon_new_node = compute_epsilon_new_node(p_new_node,temperature)

        self.max_query_count = max_query_count
        while self.now_simulation_count < simulation_count:

            print(f"[process({self.process_id})]simultation {self.now_simulation_count}")
            '''
            执行一次模拟，从根节点出发
            '''
            now_node = self.tree.root
            def check_any_possible(now_node):
                for child in now_node.children:
                    temp_node = child
                    while temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                        temp_node = temp_node.children[0]
                    if do_24(temp_node.io_state.now_datas)>0:
                        return True
                return False
            if (not check_any_possible(now_node)) and len(now_node.children) >= max_child_count:
                return 0
            
            randomly_go_to_terminal_count = 0
            first_time = True
            while len(now_node.children) > 0:
                '''
                有儿子节点，在每个地方都决定是去扩展新节点还是选择已有节点
                '''
                # decision = self.make_decision(now_node)
                if first_time:
                    decision = self.make_decision_by_value(now_node, epsilon_new_node,max_child_count,temperature)
                    first_time = False
                else:
                    decision = self.make_decision_by_value(now_node, epsilon_new_node=0.0, max_child_count=max_child_count,temperature=temperature)
                if decision == "early_stopping":
                    return 0

                assert now_node.node_type == "Action Input"
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
                if randomly_go_to_terminal_count > 100: #连续100次走不出新路径，说明搜索结束，退出
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
                    return 1


                '''
                针对candidate投票
                '''
                if self.now_simulation_count % matching_interval == 0 and len(self.terminal_node) >= 2:
                    LLM_rank_args = {
                        "functions": self.io_func.functions,
                        "process_id": self.process_id,
                        "task_description": self.io_func.task_description,
                        "input_description": self.io_func.input_description,
                        "rank_func": rank2_allchain,
                        # "rank_func": rank2_rule,
                        # "rank_func": rank2_oracle_rule,
                    }
                    if "oracleRule" in subfix:
                        LLM_rank_args["rank_func"] = rank2_oracle_rule
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
        按照推导出的ELo积分公式选择子节点
        如果选了扩展新节点，返回-1。否则返回子节点编号
        同样，不选择已经标记为finish的节点
        不要选择之前filter筛出的"新节点"
        '''
        for child in now_node.children:
            assert child.expand_num != 0
        elos = [-10000 if (child.expand_num == 0 or child.finished) else child.Elo for child in now_node.children] 
        if len(now_node.children) < max_child_count:
            elos.append(epsilon_new_node)
        temperature  = now_node.compute_choice_temperature(temperature)
        weights = softmax_bias(elos,temperature)
        if self.process_id == 0:
            print(f"Elo: ",end="")
            for elo, child in zip(elos, now_node.children):
                temp_node = child
                while temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                    temp_node = temp_node.children[0]
                print(f"{elo:.2f}({do_24(temp_node.io_state.now_datas)})",end=" ")
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
        适用于0613模式
        filter_size：代表生成子节点时一次生成多个，根据某种规则筛选最好的向下走步
        '''
        assert (not now_node.is_terminal) and (not now_node.pruned)
        assert now_node.messages != []


        while True:
            
            '''
            如果有多个子节点，先看看是否都被访问过了。有没被访问过的就访问那个
            '''
            if len(now_node.children) > 0:
                for k, child in enumerate(now_node.children):
                    if child.expand_num == 0: #没被访问过
                        assert False
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
                            now_node.pruned = True #链条过长被剪枝了

                        if now_node.is_terminal or now_node.pruned:
                            return now_node

            '''
            进入这段逻辑说明要么没孩子，要么所有孩子都被访问过了，即需要新造孩子节点
            每次都是新造filter个孩子，即使原来已经有 filter*N个孩子
            '''
            new_generated_list = []
            for _ in range(filter_size):
                '''
                如果节点有儿子节点，就拼接DIVERSITY_PROMPT，预期生成不一样的儿子节点。注入生成完message以后要再丢掉
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
                            former_candidates_des = former_candidates_des
                        diverse_prompt = diverse_prompt.replace("{previous_candidate}",former_candidates_des)
                        if temp_now_node.father == None:
                            diverse_prompt = diverse_prompt.replace("{now_observation}", self.io_func.input_description)
                        else:
                            diverse_prompt = diverse_prompt.replace("{now_observation}",temp_now_node.observation)
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
                如果拼接了diversity prompt，要去掉
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

                    if status != 0: # 错误，需要剪枝
                        # 0代表正常返回
                        # 1代表没有对应api名字
                        # 2代表输入有错误
                        # 3代表生成结束，出现final answer
                        # 4代表模型自己决定剪枝give_up
                        # ...
                        if status == 4:
                            if temp_now_node.get_depth() >= 6:
                                temp_now_node.make_finish(2)
                            else:
                                temp_now_node.make_finish(1)
                            temp_now_node.pruned = True
                        elif status == 3: #生成final_answer
                            temp_now_node.make_finish(2)
                            temp_now_node.is_terminal = True
                        elif status == 1: #出现幻觉函数名，不改的话后面的message会报错。都会出错
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
            从new_generated_list中选一个最好的来扩展，暂时没有先验方法
            '''
            assert len(new_generated_list) > 0
            if len(new_generated_list) > 1:
                '''
                给生成的next_tree_split_nodes节点们进行排序
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
                zip_value.sort(key=lambda x: x[0].prior_score, reverse=True) #先做score高的
                new_generated_list,filtered_order = zip(*zip_value)
                if self.process_id == 0:
                    print(f"scores={scores}, filtered order: {filtered_order}")

                select_child = new_generated_list[filtered_order[0]]
            else:
                select_child = new_generated_list[0]


            '''
            顺序访问最终选中的路径
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
                now_node.pruned = True #链条过长被剪枝了

            if now_node.is_terminal or now_node.pruned:
                return now_node

