import http.server
import socketserver
import json
import threading
from AgentBench.webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
import traceback
from termcolor import colored


host = "localhost"
port = 12348

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    env = WebAgentTextEnv(observation_mode="text", human_goals=True)
    def do_POST(self):
        if self.path == "/api":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            post_data_str = post_data.decode('utf-8')
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
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = message
            response_json = json.dumps(response)
            
            self.wfile.write(response_json.encode('utf-8'))
        else:
            self.send_error(404)
    def exec_action(self, idx, action_list):
        '''
        Execute action sequence in the environment and return execution results (observation, available_actions)
        '''
        MyRequestHandler.env.reset(idx)
        observation, available_actions = (MyRequestHandler.env.observation, MyRequestHandler.env.get_available_actions())
        reward = 0.0
        info = None
        done = 0

        for act in action_list:
            print(f'perform action: {act}')
            if act.startswith("search"):
                if not available_actions['has_search_bar']:
                    print(colored(f"Failed to {act}!!! Available: {available_actions}", color="magenta"))
                    # raise TypeError('no search bar')
            elif act.startswith('click'):
                product = act[6:-1]
                if str(product).lower() not in [str(action) for action in available_actions['clickables']]:
                    print(colored(f"Failed to {act}!!!Available: {available_actions}", color="magenta"))
                    # raise TypeError('no product name')
            observation, reward, done, info = MyRequestHandler.env.step(act)
            available_actions = MyRequestHandler.env.get_available_actions()
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
    with socketserver.TCPServer((host, port), MyRequestHandler) as server:
        print(f"Server started at http://{host}:{port}")
        server.serve_forever()

server_list = []
# for port in ports:
server = threading.Thread(target=run, args=(host, port))
server.start()
server_list.append(server)