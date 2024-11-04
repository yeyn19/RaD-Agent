'''
querycandidate
ELo
'''

import os
import threading
import traceback
from termcolor import colored
from Prompts.rank_prompts import LLM_PAIRWISE_RANK_ALLCHAIN_SYSTEM_PROMPT,LLM_PAIRWISE_RANK_SUBFIX_SYSTEM_PROMPT, LLM_PAIRWISE_RANK_USER_PROMPT, LLM_PAIRWISE_RANK_ALLCHAIN_LYX_SYSTEM_PROMPT, LLM_PAIRWISE_RANK_ALLCHAIN_LYX_USER_PROMPT, LYX_VOTE_FUNCTION
import random
from Tree.Tree import tree_node
from copy import deepcopy
import json
import math

from run_webshop.env_utils import parse_json


def rank2symmetry(llm_interface, LLM_rank_args, cand1,cand2):
    '''
    llm
    '''

    
    single_rank_func = LLM_rank_args["rank_func"]
    score = [0,0]
    bigger1,query_count1, total_tokens1 = single_rank_func(llm_interface, LLM_rank_args, cand1,cand2)
    if bigger1 == 0 or bigger1 == 1:
        score[1 - bigger1] += 1
    bigger2,query_count2, total_tokens2 = single_rank_func(llm_interface, LLM_rank_args, cand2,cand1)
    if bigger2 == 0 or bigger2 == 1:
        score[bigger2] += 1
    if score[0] > score[1]:
        return 1 , query_count1 + query_count2, total_tokens1 + total_tokens2
    elif score[0] < score[1]:
        return -1, query_count1 + query_count2, total_tokens1 + total_tokens2
    else:
        return 0, query_count1 + query_count2, total_tokens1 + total_tokens2

class rank2Thread(threading.Thread):
    def __init__(self, llm_interface, LLM_rank_args, system_message, user_message):
        threading.Thread.__init__(self)
        self.llm_interface = llm_interface
        self.LLM_rank_args = LLM_rank_args
        self.system_message = system_message
        self.user_message = user_message
        self.result = 0, 0, 0
    def run(self):
        self.result = rank2_single(self.llm_interface, self.LLM_rank_args, self.system_message, self.user_message)

def rank2_single(llm_interface, LLM_rank_args, system_message, user_message):
    llm_interface.change_messages([{"role":"system","content":system_message},
                                    {"role":"user","content":user_message},
                                    ])
    # llm_interface.display_conversation()
    functions = deepcopy(LLM_rank_args["functions"])
    functions.append(LYX_VOTE_FUNCTION)
    output,error_code,total_tokens = llm_interface.parse(functions=functions,function_call={"name":"choose_preference"}, process_id=LLM_rank_args["process_id"])
    # print(output)
    if "function_call" in output.keys():
        try:
            arguments = output["function_call"]["arguments"]
            prefer_key = "preference"
            try:
                arguments = json.loads(arguments)
            except BaseException as e:
                print(traceback.format_exc(), '\n', str(e))
                arguments = parse_json(arguments, prefer_key)
            if prefer_key not in arguments.keys():
                raise ValueError("Failed to parse argument")
            prefer = 1 - int(arguments[prefer_key]) #preference=01
            # print(colored(output, color='green'))
            # with open(os.path.join("compare_history.jsonl"), "a") as fw:
            #     json.dump({
            #             'cand1': cand1_des,
            #             'cand2': cand2_des,
            #             'output': output
            #         }, fw)
            #     fw.write('\n')
            if prefer != 0 and prefer != 1:
                prefer = 0
            if prefer == 1:
                return 0, 1, total_tokens
            elif prefer == 0:
                return 1, 0, total_tokens
            else:
                return 0, 0, total_tokens
            # return prefer, 1, total_tokens
        except Exception as e:
            traceback.print_exc()
            print(e)
            with open(os.path.join("error.txt"), "a") as fa:
                fa.write(str(e) + '\n')
            # llm_interface.display_conversation()
            print(colored(output, color='red'))
            # return 0.5, 1, total_tokens
            return 0, 0, total_tokens
    else:
        print(colored(output, color='red'))
        print("no function call in rank candidate")
        # return 0.5, 1, total_tokens


def rank2_allchain(llm_interface,LLM_rank_args, cand1,cand2):
    '''
    cand1,
      query
    1.LLM_rank_args
    2.multitool-multiapi-ETS
    '''
    
    print(colored("run rank 2 all chain", color='red'))

    

    system_message = LLM_PAIRWISE_RANK_ALLCHAIN_LYX_SYSTEM_PROMPT
    user_message =  LLM_PAIRWISE_RANK_ALLCHAIN_LYX_USER_PROMPT
    user_message = user_message.replace("{task_description}", LLM_rank_args["task_description"])
    user_message = user_message.replace("{input_description}", LLM_rank_args["input_description"])
    cand1_des = cand1.get_former_trice_from_this_node()
    cand2_des = cand2.get_former_trice_from_this_node()
    user_message = user_message.replace("{candidate_A}",cand1_des)
    user_message = user_message.replace("{candidate_B}",cand2_des)

    count_0 = 0
    count_1 = 0

    final_total_tokens = 0
    rank_thread_list = []
    for _ in range(3):
        rank_thread = rank2Thread(llm_interface, LLM_rank_args, system_message, user_message)
        rank_thread.start()
        rank_thread_list.append(rank_thread)
    for rank_thread in rank_thread_list:
        rank_thread.join()
        single_0, single_1, total_tokens = rank_thread.result
        count_0 += single_0
        count_1 += single_1
        final_total_tokens += total_tokens
        
    if count_0 > count_1:
        return 0, 1, final_total_tokens
    elif count_0 < count_1:
        return 1, 1, final_total_tokens
    else:
        return 0.5, 1, final_total_tokens

    
def rank2_allchain_candidate_list(llm_interface,LLM_rank_args, cand1,cand2):
    '''
    cand1,candidates
      query
    1.LLM_rank_args
    2.multitool-multiapi-ETS
    '''

    print("run candidate list")

    if cand1["cont"] == None and cand2["cont"] == None:
        return 0.5, 1, 0
    elif cand1["cont"] == None:
        return 0, 1, 0
    elif cand2["cont"] == None:
        return 1, 1, 0
    def node_list_to_former_trice(node_list):
        output_str_list = []

        for node in node_list:
            now_node_des_list = []
            now_node_des_list.append(f"{node['node_type']}: {node['description']}\n")
            if "observation" in node.keys() and node["observation"] != "":
                now_node_des_list.append(f"observation: {node['observation']}\n")
            output_str_list = output_str_list + now_node_des_list
        now_str = ""
        for k, cont in enumerate(output_str_list):
            now_str += f"step_{k+1}: {cont}\n"

        if now_str == "":
            now_str = "None"
        return now_str


    system_message = LLM_PAIRWISE_RANK_ALLCHAIN_LYX_SYSTEM_PROMPT
    user_message =  LLM_PAIRWISE_RANK_ALLCHAIN_LYX_USER_PROMPT
    user_message = user_message.replace("{task_description}", LLM_rank_args["task_description"])
    user_message = user_message.replace("{input_description}", LLM_rank_args["input_description"])
    cand1_des = node_list_to_former_trice(cand1["cont"])
    cand2_des = node_list_to_former_trice(cand2["cont"])
    user_message = user_message.replace("{candidate_A}",cand1_des)
    user_message = user_message.replace("{candidate_B}",cand2_des)

    llm_interface.change_messages([{"role":"system","content":system_message},
                                   {"role":"user","content":user_message},
                                   ])
    # llm_interface.display_conversation()
    functions = deepcopy(LLM_rank_args["functions"])
    functions.append(LYX_VOTE_FUNCTION)
    # print(llm_interface.display_conversation())
    output,error_code,total_tokens = llm_interface.parse(functions=functions,function_call={"name":"choose_preference"},process_id=LLM_rank_args["process_id"])
    # print(output)
    # exit()
    if "function_call" in output.keys():
        try:
            arguments = output["function_call"]["arguments"]
            arguments = json.loads(arguments)
            prefer = 1 - int(arguments["preference"]) #preference=01
            if prefer != 0 and prefer != 1:
                prefer = 0
            return prefer, 1, total_tokens
        except Exception as e:
            print(e)
            # llm_interface.display_conversation()
            print(output)
            return 0.5, 1, total_tokens
    else:
        print(output)
        print("no function call in rank candidate")
        return 0.5, 1, total_tokens

def rank2_subfix(llm_interface,LLM_rank_args, cand1,cand2):
    '''
    candidateprefix
    '''
    anscestor_interesction = tree_node.find_ancestor_intersection(cand1,cand2)
    assert anscestor_interesction != None
    intersect_trice = anscestor_interesction.get_former_trice_from_this_node(end_node=None)
    trice_1 = cand1.get_former_trice_from_this_node(end_node=anscestor_interesction)
    trice_2 = cand2.get_former_trice_from_this_node(end_node=anscestor_interesction)

    system_message = LLM_PAIRWISE_RANK_SUBFIX_SYSTEM_PROMPT
    system_message = system_message.replace("{task_description}", LLM_rank_args["task_description"])
    system_message = system_message.replace("{input_description}", LLM_rank_args["input_description"])
    system_message = system_message.replace("{intersect_trice}", intersect_trice)
    system_message = system_message.replace("{candidate_A}",trice_1)
    system_message = system_message.replace("{candidate_B}",trice_2)
    llm_interface.change_messages([{"role":"system","content":system_message},
                                   {"role":"user","content":LLM_PAIRWISE_RANK_USER_PROMPT},
                                   ])
    # llm_interface.display_conversation()
    # exit()
    output,error_code, total_tokens = llm_interface.parse(functions=LLM_rank_args["functions"],function_call="none",process_id=LLM_rank_args["process_id"])
    # print(output)
    # exit()
    if output["content"].strip().lower()[-1] == "a":
        return 1, 1, total_tokens
    else:
        return 0, 1, total_tokens
    
def sum_based_rankn(llm_interface,LLM_rank_args, candidates):
    '''
    pairwise
    '''
    total_querys = 0
    total_tokens = 0
    rank_details = []
    scores = [0]*len(candidates)
    for i in range(len(candidates)-1):
        for j in range(i+1,len(candidates)):
            pairwise_rank,query_count,rank2_tokens = rank2symmetry(llm_interface,LLM_rank_args, candidates[i],candidates[j])
            total_querys += query_count
            total_tokens += rank2_tokens
            if pairwise_rank > 0:
                scores[i] += 1
            elif pairwise_rank < 0:
                scores[j] += 1
            else:
                scores[i] += 0.5
                scores[j] += 0.5
            rank_details.append({"a":i,"b":j,"resualt":pairwise_rank})
    return scores, total_querys, total_tokens, rank_details


def elo_match(llm_interface, LLM_rank_args,balence_func, Elo_args, candidates,id0,id1):
    win,query_count,total_tokens = rank2symmetry(llm_interface, LLM_rank_args, candidates[id0],candidates[id1])
    if win == 0:
        win = 0.5
    elif win == -1:
        win = 0


    temperature = Elo_args["temperature"]
    expect_win_rate = 1 / ( 1 + math.e**(- (candidates[id0].Elo-candidates[id1].Elo) /temperature)  )

    delta_elo = Elo_args["k"] * (win - expect_win_rate)
    candidates[id0].Elo += delta_elo
    candidates[id1].Elo += -delta_elo

    '''
    
    '''
    now_node = candidates[id0]
    while now_node != None:
        now_node.matching_time += 1
        now_node = now_node.father
    now_node = candidates[id1]
    while now_node != None:
        now_node.matching_time += 1
        now_node = now_node.father

    '''
    
    '''
    balence_func()

    # print(f"race_result: {win}, new_elo: ",end="")
    # for cont in candidates:
    #     print(f"{cont.Elo:.2f} ",end="")
    # print()
    return query_count, total_tokens



def get_best_N(llm_interface, LLM_rank_args, candidates, N):
    '''
    candidatesN
    1.
    2.N
    
    '''
    assert N <= len(candidates)
    total_query_count, total_token_usage = 0,0
    best_N_id = []
    for _ in range(N):
        now_best_id = -1
        for i in range(1,len(candidates)):
            if i in best_N_id:
                continue
            if now_best_id == -1:
                now_best_id = i
                continue
            win,query_count,total_tokens = rank2symmetry(llm_interface, LLM_rank_args, candidates[now_best_id],candidates[i])
            total_query_count += query_count
            total_token_usage += total_tokens
            if win == -1:
                now_best_id = i
        if now_best_id != -1:
            best_N_id.append(now_best_id)
    return best_N_id, total_query_count, total_token_usage

def elo_rank(llm_interface, LLM_rank_args, candidates, new_candidate_pos,balence_func, Elo_args,root_node = None):
    '''
    elo
    LLM_rank_args:
        pairwise
    Elo_args:
        k: (EloElo())
        new_candidate_race_count: candidate
        global_race_count:
    '''
    total_query_count = 0
    total_tokens = 0

    if new_candidate_pos != []:
        for _ in range(Elo_args["new_candidate_race_count"]):
            '''
            
            '''
            random_new_node_id = random.randint(0,len(new_candidate_pos) - 1)
            new_node_elo = candidates[new_candidate_pos[random_new_node_id]].Elo
            all_elos = list(zip(range(len(candidates)), [cand.Elo for cand in candidates]))
            all_elos.sort(key = lambda x: abs(x[1] - new_node_elo)) #
            candidate_ids, _ = zip(*all_elos)
            nearest_old_id = None
            nearest_new_id = None
            for candidate_id in candidate_ids: #
                if candidate_id == random_new_node_id:
                    continue
                if candidate_id in new_candidate_pos:
                    if nearest_new_id == None:
                        nearest_new_id = candidate_id
                    continue
                nearest_old_id = candidate_id
                break
            
            id1 = nearest_new_id if nearest_old_id == None else nearest_old_id
            if id1 == None:
                continue
            id0 = new_candidate_pos[random_new_node_id]
            temp_query_count, temp_total_tokens = elo_match(llm_interface, LLM_rank_args,balence_func, Elo_args, candidates,id0,id1)
            total_query_count += temp_query_count
            total_tokens += temp_total_tokens

    for _ in range(Elo_args["global_race_count"]):
        if Elo_args["global_selction_method"] == "random": #
            r = list(range(len(candidates)))
            random.shuffle(r)
            id0,id1 = r[0], r[1]
        elif Elo_args["global_selction_method"] == "annealing":
            assert root_node != None
            node1, node2 = None, None
            count = 0
            while node1 == node2:
                if count > 0:
                    print("select the same node1 and node2")
                node1 = root_node.randomly_select_to_terminal_node(temperature=Elo_args["temperature"])
                '''
                Elo-100000node2
                '''
                temp_Elo = node1.Elo
                node1.Elo = -100000
                balence_func()
                node2 = root_node.randomly_select_to_terminal_node(temperature=Elo_args["temperature"])
                node1.Elo = temp_Elo
                balence_func()
                count += 1
            assert node1 in candidates
            assert node2 in candidates
            id0 = candidates.index(node1)
            id1 = candidates.index(node2)

        else:
            raise NotImplementedError
        temp_query_count, temp_total_tokens = elo_match(llm_interface, LLM_rank_args,balence_func, Elo_args, candidates,id0,id1)
        total_query_count += temp_query_count
        total_tokens += temp_total_tokens

    return candidates, total_query_count, total_tokens
    

def quick_sort_rank(candidates):
    '''
    LLM ,
    '''
    if len(candidates) <= 1: #
        return candidates
    pos = random.randint(0,len(candidates)-1)
    left,right = [], []
    for k in range(len(candidates)):
        if k == pos:
            continue
        out = rank2symmetry(candidates[pos],candidates[k])
        if out > 0:
            left.append(candidates[k])
        else:
            right.append(candidates[k])

    return quick_sort_rank(left) + [candidates[pos]] + quick_sort_rank(right)


if __name__ ==  "__main__":
    random.seed(42)
    # candidates = [
    #     "234",
    #     "66.5",
    #     "77.1",
    #     "88.967",
    #     "pi",
    #     # "e",
    #     # "ln(2)"
    # ]
    candidates = [
        "77.1",
        "88.967",
        "pi",
        "66.5",
        "234",
        "ln(2)"
    ]
    # output = quick_sort_rank(candidates)
    '''
    starting_delta:
    50 -> 42.85%
    100 -> 35.99%
    150 -> 29.66%
    200 -> 24.03%
    '''
    output = elo_rank(candidates)
    print(output)
