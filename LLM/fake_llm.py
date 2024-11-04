'''
gpt-turbo
token

'''
import requests
import json
import time
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import json
import openai
from LLM.openai_0613 import chat_completion_request


class chatGPT:
    def __init__(self):

        self.headers = {
            "Content-Type": "application/json"
        }


        self.proxies = {
        "http": "gfw.in.zhihu.com:18080",
        # "https": "http://127.0.0.1:10808",
        }

        self.total_tokens = 0
        self.time = time.time()

        self.memory = {}


    def generation(self, system,user,stop=None,model="text-davinci-003",**args):
        messages =  [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        output = chat_completion_request(messages, functions=None, model="gpt-4o",stop=stop, **args)
        # print(output.json())
        assistant_message = output.json()["choices"][0]["message"]["content"]
        return assistant_message
    
    def generation_(self, system,user,stop=None,model="text-davinci-003",**args):      
        '''
        
        '''
        # if self.memory.get((system,user),-1) != -1:
        #     return self.memory[(system,user)]
        while time.time() - self.time < 5:
            continue
        self.time = time.time()

        if system == "":
            system = "You are a user what to consult the assistant."
        if model == "text-davinci-003":
            payload = {
                "model": "text-davinci-003",
                "prompt": user,
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "best_of": 3,
                "stop": stop,
                **args
            }
            url =  "http://47.254.22.102:8989/query"
        elif model == "gpt-3.5-turbo":
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 1024,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "best_of": 3,
                "stop": stop,
                **args,
            }
            url =  "http://47.254.22.102:8989/chat"

        else:
            raise NotImplementedError
        
        if model != "gpt4":
            try_count = 0
            while True:
                try:

                    response = requests.post(url, json=payload, headers=self.headers,timeout=60)#,proxies=self.proxies)
                    json_data = json.loads(response.text)
                except Exception as e:
                    print(e)
                    try_count += 1
                    time.sleep(5)
                if try_count == 0 or try_count >= 3:
                    break
            
            if try_count > 3:
                return "Time out"

        try:

            if model in ["gpt4","gpt-3.5-turbo"]:
                result = json_data["choices"][0]["message"]["content"]
            elif model == "text-davinci-003":
                result = json_data["choices"][0]["text"]
            self.memory[(system,user)] = result
            print(f"total tokens: {json_data['usage']['total_tokens']}")
            return result   
        except Exception as e:
            return f"{e}: {response.text}"
                                

class GPT4(LLM):
    
    n: int = 0

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt, stop: Optional[List[str]] = None) -> str:
        
        openai.organization = ""
        openai.api_key = ""

        if isinstance(prompt, str):
            message = [
                {"role": "system", "content": "You are a user what to consult the assistant."},
                {"role": "user", "content": prompt},
            ]
        else:
            message = prompt

        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=1,
        top_p=1,
        n=1,
        max_tokens=1000,
        stop=stop,
        )
        response = json.loads(str(response))
        output = response["choices"][0]["message"]["content"]
        print(output)

        print("\n--------------------")
        print(prompt)
        print("\n********************")
        print(response)
        print(output)
        print("\n--------------------")
        input()

        return output
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
    


if __name__ == "__main__":
#     x = chatGPT()
#     prompt = '''
#     
# In this paper, we propose a new paradigm RRHF which can be tuned as easily as fine-tuning and
# achieve a similar performance as PPO in HH dataset. We also train Wombat by learning from
# ChatGPT outputs within only 2 hours. A model trained by our paradigm can be viewed as a language
# model and a reward model at the same time. Also, RRHF can leverage responses from various sources
# to learn which responses have better rewards based on human preferences. We hope this work can
# open the way to align human preferences without using complex reinforcement learning.
#     '''
#     print(x.generation("",prompt,model="gpt-3.5-turbo"))
#     print(x.total_tokens)

    x = chatGPT()
    print(x.generation("","can you combine [4,5,6,10] with only mathmatical operations to get 24? For example, you can combine [1,1,4,6] with 1*1*4*6=24.",model="gpt4"))