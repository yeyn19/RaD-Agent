from __future__ import annotations
import random
import math
import os
from typing import List, Tuple, Optional
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from openai import BadRequestError, OpenAI
from openai import APIConnectionError

from ets_utils import CODE_DIR

def do_24(list_data,stack=[]):
    '''
    递归计算，减少一个数字
    '''
    if len(list_data) == 1:
        if list_data[0] == 24:
            # print(stack)
            return True
        else:
            return False

    '''
    随机选择两个进行组合
    '''        
    for x1 in range(len(list_data) - 1):
        for x2 in range(x1+1,len(list_data)):
            d1 = list_data[x1]
            d2 = list_data[x2]
            new_data_list = [d1+d2, d1-d2, d2 - d1, d1*d2]
            if d1 != 0:
                new_data_list.append(d2/d1)
            if d2 != 0:
                new_data_list.append(d1/d2)
            for new_data in new_data_list:
                temp_stack = stack.copy()
                temp_stack.append(f"{d1},{d2}->{new_data}")
                new_list = list_data[:x1] + list_data[x1+1:x2] + list_data[x2+1:] + [new_data]
                if do_24(new_list,temp_stack):
                    return True

    return False

class env24():
    def __init__(self, nums: List, start_list: List):
        self.now_list = nums
        self.start_list = start_list

    def check_succ(self):
        if len(self.now_list) > 1:
            return False
        return abs(self.now_list[0] - 24) < 1e-3
    def get_possible_actions(self) -> List[Tuple['env24', str]]:
        actions = []
        n = len(self.now_list)

        for i in range(n):
            for j in range(i + 1, n):
                num1, num2 = self.now_list[i], self.now_list[j]
                new_lists = [
                    [num1 + num2] + [self.now_list[k] for k in range(n) if k != i and k != j],
                    [num1 - num2] + [self.now_list[k] for k in range(n) if k != i and k != j],
                    [num2 - num1] + [self.now_list[k] for k in range(n) if k != i and k != j],
                    [num1 * num2] + [self.now_list[k] for k in range(n) if k != i and k != j]
                ]

                # Avoid division by zero
                if num2 != 0:
                    new_lists.append([num1 / num2] + [self.now_list[k] for k in range(n) if k != i and k != j])
                if num1 != 0:
                    new_lists.append([num2 / num1] + [self.now_list[k] for k in range(n) if k != i and k != j])

                # Create new env24 instances and add them to actions along with the description of the action
                for new_list in new_lists:
                    new_env = env24(new_list, deepcopy(self.start_list))
                    action_description = f"{num1} and {num2} with new list {new_list}"
                    actions.append((new_env, action_description))

        return actions

def get_reward(node: Node) -> float:
    now_value = 0
    while node:
        if do_24(node.env.now_list):
            now_value += 1
        node = node.parent
    return now_value / 4

# def get_reward(node: Node) -> float:
#     return 1- np.abs(np.mean(node.env.now_list) - 24)/1000

# def get_reward(node: Node) -> float:
#     return random.random()

prompt = """Give Score from [0,1] based on who is closer to combine with +-*/ to 24 in a twenty-four game, more possible, the score is higher. Reponse only a number, don't say anything else:
numbers: 1, 2, 4, 6 -> scores: 0.3
numbers: 1, 2, 24 -> scores: 0.5
numbers: 1, 24 -> scores: 0.7
numbers: 24 -> scores: 1.0
numbers: 23 -> scores: 0.0
numbers: 20 -> scores: 0.0
numbers: 20 4 -> scores: 0.7
numbers: 20 5 -> scores: 0.2
numbers: 20 5 1 -> scores: 0.5
numbers: <<>> -> scores: 
"""
def get_reward(node):
    now_nums = node.env.now_list
    now_nums_str = " ".join([str(num) for num in now_nums])
    client = OpenAI(
        api_key="", # your api key
        base_url="", # your base url
        timeout=30,
    )
    new_prompt = prompt.replace("<<>>",now_nums_str)
    openai_model = "gpt-3.5-turbo-16k"
    openai_response = client.chat.completions.create(
        model=openai_model,
        messages = [{"role":"system","content": new_prompt}],
    )
    score = openai_response.choices[0].message.content
    try:
        return eval(score)
    except:
        print(f"error parsing: {now_nums}")
        return 0.0
    
class Node:
    def __init__(self, value: float, env, parent: Optional['Node'] = None):
        self.value = value  # 当前节点的值
        self.env = env  # 到达当前节点的运算历史
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.reward = 0  # 奖励值
        self.depth = 0
        self.done = False

    def is_fully_expanded(self):
        # 检查节点是否已经完全展开
        return len(self.children) == len(self.env.get_possible_actions())

    def best_child(self, exploration_constant):
        # 选择最佳子节点
        best_score = -float('inf')
        best_children = []
        for child in self.children:
            exploit = child.reward / child.visits
            # print(self.visits, child.visits)
            explore = math.sqrt(2.0 * math.log(self.visits) / child.visits)
            score = exploit + exploration_constant * explore
            if score == best_score:
                best_children.append(child)
            elif score > best_score:
                best_children = [child]
                best_score = score
        return random.choice(best_children)
def MCTS(numbers: List[int]) -> int:
        # return best_child(root, 0)

    def tree_policy(node: Node, exploration_constant: float) -> Node:
        while not is_terminal(node):
            if node.is_fully_expanded():
                node = node.best_child(exploration_constant)
            elif len(node.children) < 2 or node.depth == 2:
                '''只剩两个数字的话，就选一个新的，不再随机探索'''
                return expand(node)
            else:
                if random.random() < 1/(len(node.children)):
                    '''有概率选best_child，否则搜索空间太大'''
                    return expand(node)
                else:
                    node = node.best_child(exploration_constant)

        return node

    def expand(node: Node) -> Node:
        # 扩展节点
        nonlocal succ
        all_actions = node.env.get_possible_actions()
        assert len(all_actions) > len(node.children)
        new_env = all_actions[len(node.children)][0]
        new_node = Node(0, new_env)
        node.children.append(new_node)
        new_node.parent = node
        new_node.depth = node.depth + 1
        if new_node.env.check_succ():
            
            succ = True
        # print(f"expand: {node.env.now_list} -> {new_node.env.now_list}")
        return new_node


    def is_terminal(node: Node) -> bool:
        # 判断节点是否是终止状态
        # 这里需要根据实际的游戏规则来实现
        return len(node.env.now_list) == 1

    def default_policy(node: Node) -> float:
        # 默认策略（模拟）
        # 这里简单使用绝对差值作为奖励
        return get_reward(node)

    def backup(node: Node, reward: float):
        # 回溯更新节点信息
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    env = env24(numbers, deepcopy(numbers))
    root = Node(0, env)
    exploration_constant = 1.41
    iterations = 40
    succ = False
    for _ in tqdm(range(iterations)):
        node = tree_policy(root, exploration_constant)
        reward = default_policy(node)
        backup(node, reward)
        if succ:
            return _
    return -1


if __name__ == "__main__":
    query_dir = os.path.join(CODE_DIR,"Downstream_tasks","24.csv")
    task_list = []
    with open(query_dir,"r") as reader:
        for k, line in enumerate(reader.readlines()[1:]):
            if k not in range(900,1000):
                continue
            question = line.split(",")[1].split(" ")
            task_list.append([int(num) for num in question])
    # print(task_list)
    pass_count = 0
    for k, task in enumerate(task_list):
        trace_num = MCTS(task)  # 这里暂时无法运行，因为需要实现一些函数
        print(f"{k} {trace_num}")
        pass_count += (trace_num != -1) and (trace_num <= 40)
    print(f"acc: {pass_count}/{len(task_list)} = {pass_count/len(task_list):.2f}")
    # 上面的代码框架已经搭建好了，但我们还需要实现一些关键函数，
