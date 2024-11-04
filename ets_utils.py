import re
import numpy as np
import math


DATA_DIR = "./output_data"
CODE_DIR = "./"
ASSET_DIR = "./assets"

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

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string)
    
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    string = string.lower()
    return string

def change_name(name):
    if name == "from":
        name = "is_from"
    if name == "class":
        name = "is_class"
    if name == "return":
        name = "is_return"
    if name == "false":
        name = "is_false"
    if name == "true":
        name = "is_true"
    if name == "id":
        name = "is_id"
    if name == "and":
        name = "is_and"
    return name


def softmax_bias(answers,temperature=1):
    answers =  np.array([(cont/temperature) for cont in answers])
    def softmax(x): #防止数值溢出
        max = np.max(x)
        return np.exp(x-max)/sum(np.exp(x-max))
    return softmax(answers)

def compute_epsilon_new_node(p_new_node,temperature):
    '''
    根据公式换算delta
    '''
    delta = temperature* math.log(p_new_node /(1-p_new_node))
    return delta




if __name__ == "__main__":
    name = change_name(standardize("[DEPRECATED] Search airports by IP address geolocation (path-style)"))
    print(name)