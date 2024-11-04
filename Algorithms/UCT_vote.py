from Tree.Tree import my_tree, tree_node
from copy import deepcopy
from Algorithms.base_search import base_search_method
from Prompts.Tree_search_prompts import DEFAULT_POLICY_PROMPT, MAKE_REFLECTION_RPOMPT,CHOSING_NODE_PROMPT,DIVERSITY_PROMPT,VOTE_BEST_PROMPT
from termcolor import colored
import numpy as np
import re

class UCT_vote(base_search_method):
    def __init__(self,input_description,llm,io_func):
        super(UCT_vote, self).__init__()
        '''
        :
        1.
        2. 
        '''

        self.input_description = input_description
        self.llm = llm
        self.io_func = io_func
        self.simulations = []
        self.restart()

    def to_json(self):
        js_obj = {
            "win": self.status == 1,
            "simulation_count": self.now_simulation_count,
            "simulations": self.simulations,
            "tree":self.tree.to_json_recursive()
        }
        return js_obj

    def restart(self):
        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)
        self.status = 0
        self.now_simulation_count = 0
        self.simulations = []
        self.terminal_node = []
        self.total_vote = 0
        self.good_vote = 0 
        pass


    def start(self,simulation_count,epsilon_new_node,choice_count,vote_candidates,vote_count,single_chain_max_step):
        '''
        epsilon_new_node:
        vote_candidatescandidate
        vote_count
        '''
        while self.now_simulation_count < simulation_count:
            print(colored("new trail start!","yellow"))
            '''
            
            '''
            this_simulation = []
            now_node = self.tree.root
            while len(now_node.children) > 0:
                '''
                
                '''
                # decision = self.make_decision(now_node)
                decision = self.make_decision_by_value(now_node, epsilon_new_node)
                
                if decision == -1:
                    print(colored("decide to make new node!","green"))
                    break
                print(colored(f"decide to go down child {decision}","green"))

                now_node = now_node.children[decision]
                this_simulation.append({"choice":decision,"new_generated":False,"score":now_node.io_state.get_score()})
                while now_node.node_type != "Action Input" and len(now_node.children) > 0:
                    now_node = now_node.children[0]
                    this_simulation.append({"choice":0,"new_generated":False,"score":now_node.io_state.get_score()})
            if now_node.is_terminal:
                print(colored(f"randomly go down to terminal nodes","green"))
            else:
                begin_default_policy_node = now_node
                end_node = self.default_policy(now_node,this_simulation,single_chain_max_step)

                self.now_simulation_count += 1
                self.simulations.append(this_simulation)
                self.terminal_node.append(end_node)

                if end_node.io_state.check_success() == 1:
                    self.status = 1
                    return 1

                '''
                
                '''
                self.make_reflection(begin_default_policy_node,end_node)


            '''
            candidate
            '''
            self.vote(choice_count,vote_candidates,vote_count)

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

            prompt = VOTE_BEST_PROMPT
            prompt.replace("{tool_names}",f"{self.io_func.tool_names}")
            prompt = prompt.replace("{task_description}",self.io_func.task_description)
            prompt = prompt.replace("{input_description}",self.input_description)

            candidates_description = ""
            for k, child_id in enumerate(choices):
                trice = self.tree.get_former_trice(self.tree.root,self.terminal_node[child_id],valid_types=["Action","Action Input","Observation"])
                candidates_description += f"<candidate_{k}>\n{trice}"
                reflection = f"Reflection: {self.terminal_node[child_id].generated_reflection.strip()}\n"
                candidates_description += reflection
                candidates_description += "*"*30 + "\n"
            prompt = prompt.replace("{candidate_description}",candidates_description)


            # reflections = ""
            # for k, child_id in enumerate(choices):
            #     reflections = reflections + f"<candidate_{k}>: {self.terminal_node[child_id].generated_reflection.strip()}\n"
            # prompt = prompt.replace("{previous_reflections}",reflections)

            print(prompt)

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
            print(real_score)


            votes = [0]*len(choices)
            vaild_votes = 0
            for i in range(vote_count):
                '''
                
                '''
                vote = self.llm("",prompt)
                print(vote)
                best_candiate_line = vote.split("\n")[-1]
                re_pattern = r"\"?candidate[ \_](\d+)\"?"
                re_result = re.findall(re_pattern,best_candiate_line.lower())
                if re_result != []:
                    assert len(re_result) == 1
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
                    value = (vote_count_this_turn / vote_candidates - 1 / vaild_votes) / np.sqrt(vaild_votes)
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

    def make_decision_oracle(self,now_node):
        '''
        
        '''
        max_pos = -1
        max_score = -1
        for k,child in enumerate(now_node.children):
            temp_node = child
            while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                temp_node = temp_node.children[0]
            if temp_node.io_state.get_score() > max_score:
                max_pos = k
                max_score = temp_node.io_state.get_score()
        if max_score <= now_node.io_state.get_score():
            return -1
        else:
            return max_pos



    def make_reflection(self,start_node,end_node):
        '''
        start_nodeend_node,start_node
        '''
        # assert len(start_node.children) == 1
        former_trice = self.tree.get_former_trice(self.tree.root, start_node)
        new_trice = self.tree.get_former_trice(start_node.children[0], end_node)
        # print("*"*50)
        # print(former_trice)
        # print("*"*50)
        # print(new_trice)

        prompt = MAKE_REFLECTION_RPOMPT
        prompt.replace("{tool_names}",f"{self.io_func.tool_names}")
        prompt = prompt.replace("{task_description}",self.io_func.task_description)
        prompt = prompt.replace("{input_description}",self.input_description)
        prompt = prompt.replace("{former_trice}",former_trice)
        prompt = prompt.replace("{new_trice}",new_trice)
        # print("*"*50)
        '''
        
        '''
        reflections = ""
        if len(self.tree.root.reflection) > 0:
            for k, ref in enumerate(self.tree.root.reflection):
                reflections = reflections + f"Reflection: {ref.strip()}\n"
        prompt = prompt.replace("{previous_reflections}",reflections)

        # print(prompt)


        reflection = self.llm("",prompt,stop=["END REFLECTION"])
        reflection = reflection.strip().replace("\n"," ")
        print(colored(reflection,"green"))
        start_node.reflection.append(reflection)
        if start_node != self.tree.root:
            self.tree.root.reflection.append(reflection)
        end_node.generated_reflection = reflection



    def default_policy(self,now_node,this_simulation,single_chain_max_step):
        '''
        react()
        '''
        assert not now_node.is_terminal

        begin_node = now_node

        first_time = True
        while now_node.get_depth() < single_chain_max_step:
            now_trice = self.tree.get_former_trice(self.tree.root,now_node)
            prefix = DEFAULT_POLICY_PROMPT
            prefix = prefix.replace("{tool_names}",f"{self.io_func.tool_names}")
            prefix = prefix.replace("{task_description}",self.io_func.task_description)
            prefix = prefix.replace("{input_description}",self.input_description)
            prefix = prefix.replace("{former_trice}",now_trice)


            '''
            
            '''
            reflections = ""
            if len(begin_node.reflection) > 0:
                for k, ref in enumerate(begin_node.reflection):
                    reflections = reflections + f"{k+1}.{ref.strip()}\n"
            prefix = prefix.replace("{previous_reflections}",reflections)


            if first_time:
                '''
                
                '''
                if len(now_node.children) > 0:
                    diverse_prompt = DIVERSITY_PROMPT
                    former_candidates_des = ""
                    if now_node.observation != "":
                        former_candidates_des = former_candidates_des + f"former observation: {now_node.observation}\n"
                    for k, child in enumerate(now_node.children):
                        temp_node = child
                        while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                            temp_node = temp_node.children[0]
                        # child_des = self.get_former_trice(child,temp_node)
                        # former_candidates_des = former_candidates_des + f"<candidate_{k+1}>\n{child_des}"
                        former_candidates_des = former_candidates_des + f"{k+1}.Action Input: {temp_node.description}\n"
                    diverse_prompt = diverse_prompt.replace("{previous_candidate}",former_candidates_des)
                    prefix = prefix + diverse_prompt
                    # print(prefix)
                first_time = False

            next_need = {
                "Thought": "\nAction: ",
                "Action": "\nAction Input: ",
                "Action Input": "\nThought: ",
            }
            next_next_need = {
                "Thought": "\nAction Input: ",
                "Action": "\nThought: ",
                "Action Input": "\nAction: ",
            }
            next_node_type = {
                "Thought": "Action",
                "Action": "Action Input",
                "Action Input": "Thought",   
            }
            prefix = prefix + next_need[now_node.node_type]
            # print(prefix)
            
            # assert now_node.node_type == "Action Input"
            stop_str = "\nObservation:"
            '''
             (Thought, Action, Action Input) 
            '''
            llm_output = self.llm("", prefix,stop=[stop_str])
            llm_output = llm_output.replace("\n\n","\n")
            # print(llm_output)

            while llm_output != "":
                if next_next_need[now_node.node_type] in llm_output:
                    now_str = llm_output.split(next_next_need[now_node.node_type])[0]
                    llm_output = next_next_need[now_node.node_type].join(llm_output.split(next_next_need[now_node.node_type])[1:])
                else:
                    now_str = llm_output
                    llm_output = ""
                temp_node = tree_node()
                temp_node.node_type = next_node_type[now_node.node_type]
                temp_node.description = now_str
                child_io_state = deepcopy(now_node.io_state)
                if temp_node.node_type == "Action Input":
                    observation, status = child_io_state.step(action_name=now_node.description, action_input=now_str)
                    temp_node.observation = observation
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 

                temp_node.father = now_node
                now_node.children.append(temp_node)
                this_simulation.append({"choice":len(now_node.children)-1,"new_generated":True,"score":temp_node.io_state.get_score()})

                if temp_node.get_depth() >= single_chain_max_step and temp_node.node_type == "Action Input":
                    temp_node.observation = temp_node.observation + "\nsequence length too long, early stopping."
                temp_node.print()

                '''
                thought
                '''
                end_tokens = ["restart", "give up", "final answer"]
                for token in end_tokens:
                    if token in now_str.lower():
                        temp_node.is_terminal = True
                
                
                now_node = temp_node

                if now_node.is_terminal == True:
                    return now_node
        return now_node