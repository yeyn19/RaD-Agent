from Tree.Tree import my_tree, tree_node
from copy import deepcopy
from Algorithms.base_search import base_search_method
from Prompts.Tree_search_prompts import DEFAULT_POLICY_PROMPT, MAKE_REFLECTION_RPOMPT,CHOSING_NODE_PROMPT,DIVERSITY_PROMPT
from termcolor import colored

class UCT_abstract(base_search_method):
    def __init__(self,input_description,llm,io_func):
        super(UCT_abstract, self).__init__()
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
        pass


    def start(self,simulation_count,single_chain_max_step):
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
                decision = self.make_decision_oracle(now_node)
                
                if decision == -1:
                    print(colored("decide to make new node!","green"))
                    break
                print(colored(f"decide to go down child {decision}","green"))

                now_node = now_node.children[decision]
                this_simulation.append({"choice":decision,"new_generated":False,"score":now_node.io_state.get_score()})
                while now_node.node_type != "Action Input" and len(now_node.children) > 0:
                    now_node = now_node.children[0]
                    this_simulation.append({"choice":0,"new_generated":False,"score":now_node.io_state.get_score()})
            
            begin_default_policy_node = now_node
            end_node = self.default_policy(now_node,this_simulation,single_chain_max_step)

            self.now_simulation_count += 1
            self.simulations.append(this_simulation)

            if end_node.io_state.check_success() == 1:
                self.status = 1
                return 1

            '''
            
            '''
            self.make_reflection(begin_default_policy_node,end_node)




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

    def make_decision(self,now_node):
        '''
        
        '''
        former_trice = self.tree.get_former_trice(self.tree.root, now_node)
        prompt = CHOSING_NODE_PROMPT
        prompt.replace("{tool_names}",f"{self.io_func.tool_names}")
        prompt = prompt.replace("{task_description}",self.io_func.task_description)
        prompt = prompt.replace("{input_description}",self.input_description)
        prompt = prompt.replace("{former_trice}",former_trice)

        '''
        
        '''
        children_des = ""
        if len(now_node.reflection) > 0:
            reflections = "Reflections:\n"
            for ref_k, ref in enumerate(now_node.reflection):
                reflections += f"{ref_k+1}.{ref}\n"
            children_des = children_des + reflections

        for child_k, child in enumerate(now_node.children):
            child_des = f"<Decision {child_k}>\n"
            next_node = child.children[0]
            while next_node.node_type != "Action Input" and len(next_node.children) > 0:
                next_node = next_node.children[0]
            child_des += self.tree.get_former_trice(child,next_node)
            children_des = children_des + child_des
        prompt = prompt.replace("{former_child_description}",children_des)
        # print("*"*50)
        # print(prompt)
        choice = self.llm("",prompt).strip()
        print(colored(choice,"red"))
        if "decision" in choice.lower():
            decision_count = int(choice.lower().split("decision")[-1].strip())
            return decision_count
        else:
            return -1


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
        # print(prompt)

        reflection = self.llm("",prompt,stop=["END REFLECTION"])
        reflection = reflection.strip().replace("\n"," ")
        print(colored(reflection,"green"))
        start_node.reflection.append(reflection)
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
                    print(prefix)
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
            print(llm_output)

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
                    observation, status = child_io_state.step(now_str)
                    temp_node.observation = observation
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 

                temp_node.father = now_node
                now_node.children.append(temp_node)
                this_simulation.append({"choice":len(now_node.children)-1,"new_generated":True,"score":temp_node.io_state.get_score()})

                temp_node.print()

                '''
                thought
                '''
                if "give up" in now_str.lower() or "final answer" in now_str.lower():
                    temp_node.is_terminal = True
                
                
                now_node = temp_node

                if now_node.is_terminal == True:
                    return now_node
        return now_node