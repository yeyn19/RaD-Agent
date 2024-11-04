

class base_env:


    def __init__(self):
        self.task_description = ""
        self.input_description = ""
        self.tool_names = []
        self.functions = []

    def restart(self):
        '''
        
        '''
        raise NotImplementedError
    
    def get_score(self):
        '''
        
        oracle()
        '''
        raise NotImplementedError

    def step(self,action,input_str):
        '''
        
         (str, )
        '''
        raise NotImplementedError
    
    def check_success(self):
        '''
        10
        '''
        raise NotImplementedError
    
    def to_json(self):
        '''
        
        '''
        raise NotImplementedError