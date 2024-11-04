TEST_PROMPT = '''
1.You are value-GPT. Please tell the following math numbers, which is higher.
2.First you may analyze the two candiates. 
3.Finally, You must exactly say which is higher in ONE word! Exactly say the word "candidate_1" or "candidate_2".
Use the format:
Analyze: xxx
Final Answer: "candidate_1" or "candidate_2"
now, here are the inputs:
candidate_1: {candidate_1}
candidate_2: {candidate_2}
'''


LLM_PAIRWISE_RANK_ALLCHAIN_SYSTEM_PROMPT = '''
You are value-GPT, which is an expert of defining which trail is better, which trail is more close to solving the task. Here is the task description:
*******************************
{{BEGIN_DESCRIPTION}}
your_task: {task_description}
your_query: {input_description}
{{END_DESCRIPTION}}
*******************************
Here are two candidates A and B, they both try to handle the task with some function calls, Their trails are as follows.
*******************************
{{CANDIDATE_A_START}}
{candidate_A}
{{CANDIDATE_A_END}}
*******************************
{{CANDIDATE_B_START}}
{candidate_B}
{{CANDIDATE_B_END}}
*******************************
remember:
1.Good trails give an result at the end, like "give_answer" or "give_up_and_restart".
2.If a trail is tuncated, and ended with a "Finish" function call, it's not a good trail.
3.Good trails' answer contain enough information to handle the task, so "I apologize, but xxxx" is not a valid answer.
'''

# 1.Good trails give an result at the end, like "give_answer" or "give_up_and_restart".
# 2.If a trail is tuncated, and ended with a "Finish" function call, it's not a good trail.
# 3.Good trails' answer contain enough information to handle the task, so "I apologize, but xxxx" is not a valid answer.

LLM_PAIRWISE_RANK_ALLCHAIN_LYX_SYSTEM_PROMPT = '''You are a helpful annotator, that help user to annotate data.'''


LLM_PAIRWISE_RANK_ALLCHAIN_LYX_USER_PROMPT = '''Giving task description and candidate answers, I want you to choose one preferred answer based on the rules. To do so, I will give you the task description that given to the models, and the candidate answers in a list for chosen. To choose the one preferred answer, you need to first analyse answers based on rules, then give the index number of the preferred answer of JSON to `choose_preference`. 

Here are the preference rules:
1. If one candidate wins and the other not, choose the candidate that wins.
2. If both candiates don't win, choose the one that doesn't make fault functions in the process.
3. Choose the candidate that is more close to success.

Here is the task description in JSON format:
*******************************
{{BEGIN_DESCRIPTION}}
your_task: {task_description}
your_query: {input_description}
{{END_DESCRIPTION}}
*******************************

Here are the candidate answers in JSON format:
*******************************
{{CANDIDATE_0_START}}
{candidate_A}
{{CANDIDATE_0_END}}
*******************************
{{CANDIDATE_1_START}}
{candidate_B}
{{CANDIDATE_1_END}}
*******************************

Now choose the preferred answer by analysing results and the rules given, return the index in range [0,1].'''

LLM_PAIRWISE_RANK_ALLCHAIN_USER_PROMPT_EAZY = '''Giving the task description and two candidate answers, you need to choose one preferred answer which is closer to success.

Here is the task description in JSON format:
*******************************
{{BEGIN_DESCRIPTION}}
your_task: {task_description}
your_query: {input_description}
{{END_DESCRIPTION}}
*******************************

Here are the candidate answers in JSON format:
*******************************
{{CANDIDATE_0_START}}
{candidate_A}
{{CANDIDATE_0_END}}
*******************************
{{CANDIDATE_1_START}}
{candidate_B}
{{CANDIDATE_1_END}}
*******************************

You should first make an analysis of the two candidates, including their strength and weakness, then you should analysis which candidate needs to be further explored. Further explore means that you can try to do similar things later on. Finally, you need to give a function call to give your preference.'''

LLM_PAIRWISE_RANK_ALLCHAIN_LYX_USER_PROMPT_OLD = '''Giving task description and candidate answers, I want you to choose one preferred answer based on the rules. To do so, I will give you the task description that given to the models, and the candidate answers in a list for chosen. To choose the one preferred answer, you need to first analyse answers based on rules, then give the index number of the preferred answer of JSON to `choose_preference`. 

Here are the preference rules:
1. if both answers give the none empty `final_answer`, check whether the given `final_answer` solves the given query.
1.1 if both answers solve the query, choose one with smaller `total_steps`.
1.1.1 if `total_steps` are same, choose one answer with better `final_answer` quality.
1.2 if one answer solve while the other not, chose the answer that solve query.
1.3 if both answers failed, check the `answer_details` to choose one with considering following preference:
1.3.1 check `response` and prefer more successful tool calling.
1.3.2 check `name` and prefer using more various tool usage.
1.3.3 prefer smaller `total_steps`.
2. if one give none empty `final_answer` while other not, choose the one give `final_answer`.
3. if both failed to give none empty `final_answer`, following 1.3 to choose one with better `answer_details`.

Here is the task description in JSON format:
*******************************
{{BEGIN_DESCRIPTION}}
your_task: {task_description}
your_query: {input_description}
{{END_DESCRIPTION}}
*******************************

Here are the candidate answers in JSON format:
*******************************
{{CANDIDATE_0_START}}
{candidate_A}
{{CANDIDATE_0_END}}
*******************************
{{CANDIDATE_1_START}}
{candidate_B}
{{CANDIDATE_1_END}}
*******************************

Now choose the preferred answer by analysing results and the rules given, return the index in range [0,1].'''


LLM_PAIRWISE_RANK_SUBFIX_SYSTEM_PROMPT = '''
You are value-GPT, which is an expert of defining which trail is better, which trail is more close to solving the task. 
All candidate tries to solve this task with some funciton calls:
*******************************
{{BEGIN_DESCRIPTION}}
your_task: {task_description}
your_query: {input_description}
{{END_DESCRIPTION}}
*******************************
First, all candidate do the following things:
{intersect_trice}
After that, there are two candidates 0 and 1, they do different things:
*******************************
{{CANDIDATE_0_START}}
{candidate_A}
{{CANDIDATE_0_END}}
*******************************
{{CANDIDATE_1_START}}
{candidate_B}
{{CANDIDATE_1_END}}
Which try do you think is more helpful to solving the task?
'''

LYX_VOTE_FUNCTION = {
    "name": "choose_preference",
    "description": "Choose the preferred answer for the query within all given answers.",
    "parameters": {
        "type": "object",
        "properties": {
            "preference": {
                "type": "number",
                "description": "The index of the preferred answer in all given answers. should be 0 or 1"
            },
        },
    },
}



LLM_PAIRWISE_RANK_USER_PROMPT = '''
Tell me which candidate is better in ONE Word: "A" or "B":'''