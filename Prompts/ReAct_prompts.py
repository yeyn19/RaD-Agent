
PREFIX = """Do the following tasks as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Task: the task you must handle
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. (or) I give up and retry.
Final Answer: the final answer to the original input question. (or) I give up and try again.
Here is the task:
{task_description}
{input_description}
Begin!
{former_trice}"""



REACT_DIVERSE_PROMPT = '''There are some former choices.
**********************************
{previous_candidate}**********************************
I will make Action that is different from all of the above.
'''


FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """You are an AutoGPT, capable of utilizing numerous tools and functions to complete the given task. 
1.First, I will provide you with the task description, and your task will commence. 
2.At each step, you need to analyze the current status and determine the next course of action by executing a function call. 
3.Following the call, you will receive the result, transitioning you to a new state. Subsequently, you will analyze your current status, make decisions about the next steps, and repeat this process. 
4.After several iterations of thought and function calls, you will ultimately complete the task and provide your final answer. 

Remember: 
1.The state changes are irreversible, and you cannot return to a previous state.
2.Keep your thoughts concise, limiting them to a maximum of five sentences.
3.You can make multiple attempts. If you plan to try different conditions continuously, perform one condition per try.
Let's Begin!
Task description: {task_description}"""


FORMAT_INSTRUCTIONS_USER_FUNCTION = """
{input_description}
Begin!
"""