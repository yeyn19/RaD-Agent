from LLM.fake_llm import chatGPT
from termcolor import colored
from Algorithms.single_chain import single_chain
from Algorithms.reflexion import reflexion_chain
from Algorithms.BFS import BFS_tree_search
from Algorithms.UCT_abstract import UCT_abstract
from Algorithms.UCT_vote import UCT_vote
from Downstream_tasks.base_env import base_env
from utils import do_24
import re
import os
from functools import partial
import json

from typing import overload


class mini_crossword(base_env):
    def __init__(self,data_list):
        super(mini_crossword, self).__init__()

        self.task_description = '''
Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters. The output looks like:
S K A L D
W A T E R
O U T R O
O R L O N
P I E T Y
1.you will get 5 vertical clues, such as "h1. Scald; an ancient Scandinavian bard".
2.you will get 5 vertical clues, such as "v3. Mine refuse".
Given the current status, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low), using the format "h1. skald (medium)" or "v2. kauri (low)". Use "certain" cautiously and only when you are 100% sure this is the correct word. 
Here is one Example:
input: 
h1. Scald; an ancient Scandinavian bard
h2. H2O; to irrigate
h3. The companion to an "intro", a postscript or exit piece
h4. An artificial fabric
h5. Deep religious feeling
v1. To rush; to stoop; a descent
v2. A New Zealand fir tree
v3. Mine refuse
v4. The garden dormouse
v5. Like a drone; humming
Thought: I can start with h1.
Action: crossword
Action Input: h1. skald (medium)
Observation: Now data
s k a l d
_ _ _ _ _
_ _ _ _ _
_ _ _ _ _
_ _ _ _ _
Action: crossword
Action Input: h2. water (medium)
Observation: Now data
s k a l d
w a t e r
_ _ _ _ _
_ _ _ _ _
_ _ _ _ _
Action: crossword
Action Input: h3. outro (medium)
Observation: Now data
s k a l d
w a t e r
o u t r o
_ _ _ _ _
_ _ _ _ _
Action: crossword
Action Input: h4. orlon (medium)
Observation: Now data
s k a l d
w a t e r
o u t r o
o r l o n
_ _ _ _ _
Action: crossword
Action Input: h5. piety (medium)
Observation: Now data
s k a l d
w a t e r
o u t r o
o r l o n
p i e t y
You win.
''' 

        self.input_description = ""
        self.tool_names = ["crossword"]

        clues,words = data_list
        words = [w.lower() for w in words]
        self.h_clue = clues[:5]
        self.v_clue = clues[5:]
        self.words = []
        for i in range(5):
            self.words.append(words[i*5:(i+1)*5])

        self.make_input_description()

        self.restart()


    def make_input_description(self):

        self.input_description = '''here is your input clues:
'''
        for k, clue in enumerate(self.h_clue):
            self.input_description = self.input_description + f"h{k+1}.{clue}\n"
        for k, clue in enumerate(self.v_clue):
            self.input_description = self.input_description + f"v{k+1}.{clue}\n"

    def restart(self):
        self.status = 0
        self.now_datas = [["_","_","_","_","_"] for _ in range(5)]


    def check_success(self):
        if not self.is_full():
            return 0
        s = self.get_score()
        if s == 25:
            return 1
        else:
            return -1

    def get_score(self):
        score = 0
        for i in range(5):
            for j in range(5):
                if self.now_datas[i][j] == self.words[i][j]:
                    score += 1
        word_suc = 0
        for i in range(5):
            temp = True
            for j in range(5):
                if self.now_datas[i][j] != self.words[i][j]:
                    temp = False
                    break
            if temp:
                word_suc += 1
            temp = True
            for j in range(5):
                if self.now_datas[j][i] != self.words[j][i]:
                    temp = False
                    break
            if temp:
                word_suc += 1
        result = {
            "score":score,
            "letter_rate": score / 25,
            "word_rate": word_suc / 10,
            "win": score == 25,
        }
        return result

    def to_json(self):
        now_words = []
        for cont in self.now_datas:
            now_words.append("".join(cont))
        js_obj = {
            "now_words": now_words,
            "status": self.status,
            "score": self.get_score(),
        }
        return js_obj
    def is_full(self):
        for words in self.now_datas:
            for word in words:
                if word == "_":
                    return False
        return True
    def get_now_state_str(self):
        state_str = "Now data:\n"
        for words in self.now_datas:
            state_str = state_str + " ".join(words) + "\n"
        return state_str

    def step(self,input_str):
        # print(input_str)
        re_pattern = r"([hv])(\d)\.[ ]?([a-zA-Z]+)[^a-zA-Z].*"
        re_result = re.match(re_pattern,input_str)
        if re_result:
            word = re_result.group(3).lower()
            if len(word) != 5:
                return f"{word} is not of 5 letters, "+self.get_now_state_str(),0
            horizon = re_result.group(1) == "h"
            id = int(re_result.group(2))
            if not (id >= 1 and id <= 5):
                return f"not a valid position {re_result.group(1)}.{id}, "+self.get_now_state_str(),0
            warn = ""
            if horizon:
                for i in range(5):
                    if self.now_datas[id-1][i] != "_":
                        warn = warn + f"(h{id-1}, v{i}) have letter {self.now_datas[id-1][i]} former and changed to {word[i]}, be careful\n"
                    self.now_datas[id-1][i] = word[i]
            else:
                for i in range(5):
                    if self.now_datas[i][id-1] != "_":
                        warn = warn + f"(h{i}, v{id-1}) have letter {self.now_datas[i][id-1]} former and changed to {word[i]}, be careful\n"
                    self.now_datas[i][id-1] = word[i]

            status = self.check_success()
            self.status = status
            if status == 1:
                return f"{warn}{self.get_now_state_str()} You Win", 1
            elif status == -1:
                return f"{warn}{self.get_now_state_str()} You Lose", -1
            else:
                return f"{warn}{self.get_now_state_str()}", 0
        return "format error, please use \"hi/vi. word (confidence) \"", 0

            

def test_all(method):
    file_name = r"C:\git_repos\LLM-AlphaGo\Downstream_tasks\raw_data\mini0505_0_100_5.json"
    output_dir = rf"C:\git_repos\LLM-AlphaGo\test_result\crossword_{method}"
    os.makedirs(output_dir,exist_ok=True)
    data_list = []
    with open(file_name,"r") as f:
        data_list = json.load(f)
    
    total = 0
    win = 0
    for k, data in enumerate(data_list):
        total += 1

        flatten_input = str(k)
        if os.path.exists(os.path.join(output_dir,f"{flatten_input}.json")):
            with open(os.path.join(output_dir,f"{flatten_input}.json"),"r",encoding="utf-8") as reader:
                json_obj = json.load(reader)
                if json_obj["win"] == True:
                    win += 1
            continue


        env, chain = test_crossword(data,method)
        if chain.status == 1:
            win += 1
            print(colored(f"You Win, ratio={win}/{total}={win/total}","green"))
        else:
            print(colored(f"You Lose, ratio={win}/{total}={win/total}","green"))

        with open(os.path.join(output_dir,f"{flatten_input}.json"),"w",encoding="utf-8") as writer:
            json_obj = chain.to_json()
            json.dump(json_obj,writer,indent=2)

def test_crossword(list_data,method):
    print(colored(f"now playing {list_data}","green"))
    env = mini_crossword(list_data)
    input_description = env.input_description
    # print(env.input_description)
    llm = chatGPT()
    llm_forward = partial(llm.generation,model="text-davinci-003")
    if method == "CoT":
        chain = single_chain(input_description=input_description,pass_at=1,llm=llm_forward,io_func=env)
        result = chain.start(single_chain_max_step=30)
    elif method == "Reflexion":
        chain = reflexion_chain(input_description=input_description, llm=llm_forward,io_func=env)
        result = chain.start(max_chain_count=3,single_chain_max_step=30)
    elif method == "BFS":
        chain = BFS_tree_search(input_description=input_description,llm=llm_forward,io_func=env)
        result = chain.start(search_width=3, expansion_ratio=3, single_chain_max_step=18,max_iters=4)
    elif method == "UCT_abstract":
        chain = UCT_abstract(input_description=input_description,llm=llm_forward,io_func=env)
        result = chain.start(simulation_count=10,single_chain_max_step=18)
    elif method == "UCT_vote":
        chain = UCT_vote(input_description=input_description,llm=llm_forward,io_func=env)
        result = chain.start(simulation_count=10,epsilon_new_node=0.3,vote_candidates=2,vote_count=3,single_chain_max_step=18)
    return env, chain


if __name__ == "__main__":
    test_all("Reflexion")