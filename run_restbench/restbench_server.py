import http.server
import os
import socketserver
import json
import threading
import traceback
from termcolor import colored
from RestGPT.test import get_api_selection, get_api_docs, get_response
from config import env_datasets_path
#  IP 
host = "localhost"
port = 12348

# for spotify test set:
# env_datasets_path = "RestGPT/datasets/spotify.json"
def get_observation(query):
    output = "You have the following api to call: \n"
    output += get_api_selection(query)
    return output

# 
class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        #  /api
        if self.path == "/api":
            # 
            content_length = int(self.headers['Content-Length'])
            # 
            post_data = self.rfile.read(content_length)
            # 
            post_data_str = post_data.decode('utf-8')
            #  JSON 
            post_data_json = json.loads(post_data_str)
            print(post_data_json)
            # function: search, click, get_available_actions, observation
            try:
                function = post_data_json['function']
                print(f"function: {function}")
                if function == 'exec_action':
                    idx = post_data_json['idx']
                    action_list = post_data_json['action_list']
                    message = self.exec_action(idx, action_list)
                    self.send_response(200)
                else:
                    message = {'error' : 'unexpected function'}
                    print(message, post_data_json)
                    self.send_response(400)
            except BaseException as e:
                print(traceback.format_exc())
                print(e)
                message = {'error' : 'invalid json'}
                self.send_response(400)
            #  Content-Type  application/json
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            #  JSON 
            response = message
            response_json = json.dumps(response)
            
            # 
            self.wfile.write(response_json.encode('utf-8'))
        else:
            #  /api 404 
            self.send_error(404)
    def exec_action(self, idx, action_list):
        '''
        action(observation, available_actions)
        '''
        with open(os.path.join(env_datasets_path), "r", encoding='utf-8') as fr:
            data_sets = json.load(fr)
            env_info = data_sets[idx]
            self.env_query = env_info['query']
        observation, available_actions = get_observation(env_info['query']), None
        
        reward = 0.0
        info = None
        done = 0
        for act in action_list:
            observation, reward, done, info = get_observation(env_info['query']), None, None, None
            available_actions = None
            if act['action'] == "request": 
                payload = act['payload']
                action = act['method']
                result = get_response(action=action, data=payload)
                observation = str(result)
            elif act['action'] == 'check_api_docs':
                api_name = act["api_name"]
                result = get_api_docs(api_name, query=self.env_query)
                observation = str(result)
        observation += f"\nYour task is: \n{self.env_query}\n Begin!"
        if done:
            print(f"reward: {reward}")
            
        return {
            'observation': observation, 
            'reward': reward, 
            'done': done, 
            'info': info, 
            'available_actions': available_actions
        }

def run(host, port):
    # 
    with socketserver.TCPServer((host, port), MyRequestHandler) as server:
        print(f"Server started at http://{host}:{port}")
        # 
        server.serve_forever()

server_list = []


server = threading.Thread(target=run, args=(host, port))
server.start()
server_list.append(server)