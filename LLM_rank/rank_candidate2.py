'''
评测一个query对应不同candidate的得分
使用ELo匹配机制？
'''

from Prompts.rank_prompts2 import LLM_PAIRWISE_RANK_ALLCHAIN_SYSTEM_PROMPT,LLM_PAIRWISE_RANK_SUBFIX_SYSTEM_PROMPT, LLM_PAIRWISE_RANK_USER_PROMPT, LLM_PAIRWISE_RANK_ALLCHAIN_LYX_SYSTEM_PROMPT, LLM_PAIRWISE_RANK_ALLCHAIN_LYX_USER_PROMPT, LYX_VOTE_FUNCTION, LLM_PAIRWISE_RANK_ALLCHAIN_USER_PROMPT_EAZY
import random
from Tree.Tree import tree_node
from copy import deepcopy
import json
import math
from ets_utils import do_24


def rank2symmetry(llm_interface, LLM_rank_args, cand1,cand2):
    '''
    使用llm比较高低，由于顺序性，需要两个各在前面比较一次
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


def rank2_oracle_rule(llm_interface,LLM_rank_args, cand1,cand2):
    data_1 = cand1.io_state.now_datas
    data_2 = cand2.io_state.now_datas
    data_1_ok = do_24(data_1)
    data_2_ok = do_24(data_2)
    # print(f"{data_1}->{data_1_ok}, {data_2}->{data_2_ok}")
    if data_1_ok and data_2_ok: #两个都ok
        return len(data_1) < len(data_2), 0, 0
    elif not data_1_ok and not data_2_ok: #两个都不行， 看看距离行差了多远
        distance_1 = 0
        node = cand1
        while node.father != None:
            node = node.father
            distance_1 += 1
            node_can = do_24(node.io_state.now_datas)
            if node_can:
                break
        distance_2 = 0
        node = cand2
        while node.father != None:
            node = node.father
            distance_2 += 1
            node_can = do_24(node.io_state.now_datas)
            if node_can:
                break
        return distance_1 < distance_2, 0, 0
    else:
        return data_1_ok, 0, 0

def rank2_rule(llm_interface,LLM_rank_args, cand1,cand2):
    data_1 = cand1.io_state.now_datas
    data_2 = cand2.io_state.now_datas
    if len(data_1) != len(data_2): #数字少的更好
        return len(data_1) < len(data_2), 0, 0
    if len(data_1) != 1: #都超过一个数字，平手
        return 0.5, 0, 0

    # 距离24更近的更好
    return math.fabs(data_1[0] - 24) <= math.fabs(data_2[0] - 24) , 0, 0
    

def rank2_allchain(llm_interface,LLM_rank_args, cand1,cand2):
    '''
    cand1在前,
    返回 左边的样本是否更好 以及query的次数 消耗的token
    1.这个函数的实现和下游任务有关，建议只修改这个函数的实现，需要的参数在瀑布上流从LLM_rank_args参数传入
    2.这份实现是multitool-multiapi-ETS算法的排序算法
    '''
    # import pdb; pdb.set_trace()
    system_message = LLM_PAIRWISE_RANK_ALLCHAIN_LYX_SYSTEM_PROMPT
    # user_message =  LLM_PAIRWISE_RANK_ALLCHAIN_LYX_USER_PROMPT
    user_message = LLM_PAIRWISE_RANK_ALLCHAIN_USER_PROMPT_EAZY
    user_message = user_message.replace("{task_description}", LLM_rank_args["task_description"])
    user_message = user_message.replace("{input_description}", LLM_rank_args["input_description"])
    cand1_des = cand1.get_former_trice_from_this_node()
    cand2_des = cand2.get_former_trice_from_this_node()
    user_message = user_message.replace("{candidate_A}",cand1_des)
    user_message = user_message.replace("{candidate_B}",cand2_des)


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
            arguments = json.loads(arguments)
            prefer = 1 - int(arguments["preference"]) #preference=0说明左边厉害，要返回1
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

    
def rank2_allchain_candidate_list(llm_interface,LLM_rank_args, cand1,cand2):
    '''
    cand1在前,专为candidates设计的
    返回 左边的样本是否更好 以及query的次数
    1.这个函数的实现和下游任务有关，建议只修改这个函数的实现，需要的参数在瀑布上流从LLM_rank_args参数传入
    2.这份实现是multitool-multiapi-ETS算法的排序算法
    '''
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
            prefer = 1 - int(arguments["preference"]) #preference=0说明左边厉害，要返回1
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
    此时假设两个candidate有一个很长的共同的prefix
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
    所有两两对做pairwise排序，把总分加和，选出最好的
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
    添加匹配值
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
    重新平衡
    '''
    balence_func()

    # print(f"race_result: {win}, new_elo: ",end="")
    # for cont in candidates:
    #     print(f"{cont.Elo:.2f} ",end="")
    # print()
    return query_count, total_tokens



def get_best_N(llm_interface, LLM_rank_args, candidates, N):
    '''
    从candidates中选出最好的N个节点
    1.算法一：小顶堆，每次和堆顶比较
    2.算法二：做N轮的冒泡排序
    复杂度一样，简单起见，选择算法二
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
    elo机制，进行多次排序
    LLM_rank_args:
        做pairwise排序用到的参数，可能和下游任务有关系
    Elo_args:
        k: 单次比赛的加减分比例(有些Elo实现上这个值可变，原来Elo越大这个就越小(收敛))
        new_candidate_race_count: 首先将新加入的candidate和别人做比赛
        global_race_count:比赛几轮
    '''
    total_query_count = 0
    total_tokens = 0

    if new_candidate_pos != []:
        for _ in range(Elo_args["new_candidate_race_count"]):
            '''
            给新节点做定位赛：先随机一个新节点，然后找到积分最接近的老节点，然后做积分
            '''
            for i in range(len(new_candidate_pos)):
                # random_new_node_id = random.randint(0,len(new_candidate_pos) - 1)
                random_new_node_id = i
                new_node_elo = candidates[new_candidate_pos[random_new_node_id]].Elo
                all_elos = list(zip(range(len(candidates)), [cand.Elo for cand in candidates]))
                all_elos.sort(key = lambda x: abs(x[1] - new_node_elo)) #小的在前面
                candidate_ids, _ = zip(*all_elos)
                nearest_old_id = None
                nearest_new_id = None
                for candidate_id in candidate_ids: #不能是自己，优先老节点，其次新节点
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
        if Elo_args["global_selction_method"] == "random": #随机选叶子节点
            r = list(range(len(candidates)))
            random.shuffle(r)
            id0,id1 = r[0], r[1]
            prob1, prob2 = 0, 0
        elif Elo_args["global_selction_method"] == "annealing":
            assert root_node != None
            node1, node2 = None, None
            count = 0
            while node1 == node2:
                if count > 0:
                    print("select the same node1 and node2")
                node1, prob1 = root_node.randomly_select_to_terminal_node(temperature=Elo_args["temperature"])
                '''
                为了选一个另外的节点，先把之前节点Elo置为-10000，那么选到的概率就是0，再接着选node2
                '''
                temp_Elo = node1.Elo
                node1.Elo = -100000
                balence_func()
                node2, prob2 = root_node.randomly_select_to_terminal_node(temperature=Elo_args["temperature"])
                node1.Elo = temp_Elo
                balence_func()
                count += 1
            assert node1 in candidates
            assert node2 in candidates
            id0 = candidates.index(node1)
            id1 = candidates.index(node2)

        else:
            raise NotImplementedError
        print(f"global match selection_rate= {prob1:.2f} - {prob2:.2f}")
        temp_query_count, temp_total_tokens = elo_match(llm_interface, LLM_rank_args,balence_func, Elo_args, candidates,id0,id1)
        total_query_count += temp_query_count
        total_tokens += temp_total_tokens

    return candidates, total_query_count, total_tokens
    

def quick_sort_rank(candidates):
    '''
    LLM 快速排序,从小到大排序
    '''
    if len(candidates) <= 1: #递归基
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
