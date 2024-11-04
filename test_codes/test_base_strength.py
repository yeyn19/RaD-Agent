from LLM.fake_llm import chatGPT



prompt = f'''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the left numbers to obtain a new number. 
Remember to use all of the provided numbers, and all of the number must be used once.
For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.
You should follow the format :
Action: play_24
Action Input: combine x, y by x "operations" y
Where x,y is the number you want to combnine now, and the Observation will tell you what numbers are left now.
Because there are 4 numbers, so you only need 3 actions to combine all numbers, you cannot undo your actions.
Now, task begin. your input is: [1,2,4,7]'''

inputs = '''Here are some candidates of the first steps, can you tell me which is better?
<candidate_1> combine 1,2, use 1+2=3, left data (3,4,7)
<candidate_2> combine 1,7, use 1+7=8, left data (2,4,8)
<candidate_3> combine 4,7, use 4*7=28, left data (1,2,28)
Tell me your choice, and tell me what next steps will you do.
'''

# llm = chatGPT()
# re = llm.generation("",prompt+inputs)
# print(re)


prompts = f'''Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
1 24
'''

# llm = chatGPT()
# re = llm.generation("",prompts)
# print(re)

vote_prompt = '''Many assistants are performing a task, And your task is to judge who's try is the best.
The Task Description is as the follows:
********************************
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the left numbers to obtain a new number.
Remember all of the number must be used , and must be used ONCE. So Only when left numbers is exact 24, you will win.
For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.
You should follow the format:
Action: play_24
Action Input: x operations y = result, (left: a b c)
Where x,y is the number you want to combnine now, and the Observation will tell you what numbers are left now.
Because there are 4 numbers, so you only need 3 actions to combine all numbers, you cannot undo your actions.
Here is one examples: [3,13,9,7]
Action: play_24
Action Input: 3*13=39 (left: 39 9 7)
Observation: 3*13=39, left: [9,7,39]
Thought: There left 3 numbers 9,7,39, they are both not the factor of 24. So the final operation must be + or -. If I use 39 in last
step, the other value need to be 24+39=63, and 63 equals to 7*9. I will combine 7 and 9.
Action: play_24
Action Input: 7*9=63 (left: 39 63)
Observation: 7*9=63, left: [63,39]
Thought: There are only 2 numbers 63,39 now. 63+39=102, 63*39>24, 63/39 is not a integer, 63-39=24. I make it!
Action: play_24
Action Input: 63-39=24 (left: 24)
Observation: 63-39=24, left: [24], you win

The real task input is: [1, 2, 4, 7]
********************************
There are some former candidates of this task
<candidate_0>
Thought: Looking at the numbers, I see that 1 and 2 are relatively small, so it might be a good idea to combine them first.
Action: play_24
Action Input: 1+2=3 (left: 3 4 7)
Observation: 1+2=3, left numbers: [4, 7, 3]
Thought: The number 3 can be combined with 4 or 7 using basic arithmetic operations. Since 4 is also a small number, I will combine it with 3.
Action: play_24
Action Input: 3*4=12 (left: 12 7)
Observation: 3*4=12, left numbers: [7, 12]
Thought: I have two larger numbers left, 7 and 12. Multiplying them won't get me closer to 24. I can try to add or subtract them instead. Since 12 is larger, I will subtract 7 from it to get 5.
Action: play_24
Action Input: 12-7=5 (left: 5)
Observation: 12-7=5, left numbers: [5], you lose
Reflection: In this try, I first followed a former trice, then tried a different approach. However, I failed to obtain the target number 24. Upon analyzing my actions, I see that I combined the smaller numbers first, which is a good strategy. However, I did not consider all possible operations to get closer to 24. In the end, I was left with two larger numbers, which are more difficult to combine. Next time, I will try to consider all possible operations and combinations.
<candidate_1>
Thought: Looking at the numbers 1, 2, 4, 7, the first step is to combine 1 and 2, which is one of the smaller numbers. The operations available are +, -, *, /. Since there are only two numbers, / is not possible. Also, 1-2 and 2-1 both result in negative numbers which are not helpful. Thus, I will combine 1 and 2 using addition.
Action: play_24
Action Input: 1+2=3 (left: 3 4 7)

Observation: 1+2=3, left numbers: [4, 7, 3]
Thought: With the number 3 combined, I will now look at the remaining numbers 4 and 7. Since they are both larger numbers, I will try to get closer to 24 by multiplying them. Therefore, I will combine the numbers 4 and 7 using multiplication.
Action: play_24
Action Input: 4*7=28 (left: 3 28)

Observation: 4*7=28, left numbers: [3, 28]
Thought: With the number 28 obtained, I will now look at the remaining number 3. Since 28 is a larger number and 3 is a smaller number, I will try to combine them using subtraction or division. Since 28 is not evenly divisible by 3, I will try subtraction. Therefore, I will combine the numbers 3 and 28 using subtraction.
Action: play_24
Action Input: 28-3=25 (left: 25)

Observation: 28-3=25, left numbers: [25], you lose
Reflection: This time, I followed the strategy of combining smaller numbers first and then larger numbers. However, the remaining numbers did not allow for easy combination. In the second step, multiplying 4 and 7 seemed like the best option, but it did not lead to
a successful combination in the last step. In hindsight, perhaps I could have tried combining 4 and 7 using division or subtraction instead. Overall, I need to be more flexible in my approach and consider all possible combinations.

When you give your vote, You must follow the following principles:
1.Better candidate's plan is sufficient and is more helpful to solve the task.
2.Better candidate is more close to the success of the task.
Begin:
1.it's your turn to first analyze each candidates. One cnadiates one sentence.
2.Then you must tell which is the best in one Word. with the following format:
analyze candidate 1:
analyze candidate 2:
...
analyze candidate n:
best candidate: "candidate x" / "tied"
Begin!
'''


vote_prompt_relative = '''I am using monte-carlo search to get closer to the success of a task, I will give you some tree path candidates, and you must tell me which candidate need to be further explored.
The Task Description is as the follows:
********************************
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. 
At first, you will be given 4 numbers(4 left numbers). 
At each step, you are only allowed to choose two of the left numbers to obtain a new number,so the left numbers decrease by 1.
Remember all of the number must be used, and must be used ONCE. So Only when left numbers is exact 24, you will win.
"Observation" will tell you the valid left numbers, you can only use two of the left numbers to combine in the next action.
For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.
You should follow the format:
Action: play_24
Action Input: x operations y = result, (left: a b c)
Where x,y is the number you want to combnine now, and the Observation will tell you what numbers are left now.
Because there are 4 numbers, so you only need 3 actions to combine all numbers, you cannot undo your actions.
Here is one examples: [3,13,9,7]
Action: play_24
Action Input: 3*13=39 (left: 39 9 7)
Observation: 3*13=39 (left: 39 9 7)
Thought: There left 3 numbers 9,7,39, they are both not the factor of 24. So the final operation must be + or -. If I use 39 in last
step, the other value need to be 24+39=63, and 63 equals to 7*9. I will combine 7 and 9.
Action: play_24
Action Input: 7*9=63 (left: 39 63)
Observation: 7*9=63 (left: 39 63)
Thought: There are only 2 numbers 63,39 now. 63+39=102, 63*39>24, 63/39 is not a integer, 63-39=24. I make it!
Action: play_24
Action Input: 63-39=24 (left: 24)
Observation: 63-39=24 (left: 24), you win

The real task input is: [1, 2, 4, 7]
There are some former candidates of this task:
<candidate_0>
Action: play_24
Action Input: 1+7=8 (left: 2 4 8)
Observation: 1+7=8 (left: 2 4 8)
Action: play_24
Action Input: 8-2=6 (left: 4 6)
Observation: 8-2=6 (left: 4 6)
Action: play_24
Action Input: 4+6=10 (left: 10)
Observation: 4+6=10 (left: 10), you lose

********************************
<candidate_1>
Action: play_24
Action Input: 1+2=3 (left: 3 4 7)
Observation: 1+2=3 (left: 3 4 7)
Action: play_24
Action Input: 4*7=28 (left: 3 28)
Observation: 4*7=28 (left: 3 28)
Action: play_24
Action Input: 28-3=25 (left: 25)
Observation: 28-3=25 (left: 25), you lose

********************************
When you give your vote, You must follow the following principles:
1.Best candidate is staying in the state which is more close to success. 
2.If you want to restart the task with one of the candidates' trice, and make some different choice in the middle, you may want to follow the best candidates' trice.
3.Remember, longer action chain doesn't mean is closer to success, because the process can not rollback, you may refer to the example as "what is  a success". 
Begin:
1.it's your turn to first analyze each candidates. One candiate one sentence.
2.Then you can tell me who do you think is most close to success, and why.
3.Finally you must tell which is the best in one Word. with the following format:
analyze candidate 1: xxx
analyze candidate 2: xxx
...
analyze candidate n: xxx
most close to succeess: xxx
best candidate: "candidate x" / "tied"
Begin!
'''

# Action: play_24
# Action Input: 4+6=10 (left: 10)
# Observation: 4+6=10, left numbers: [10], you lose

# When you give your vote, You must follow the following principles:
# 1.Best candidate is staying in the state which is more close to success. 
# 2.If you want to restart the task with one of the candidates' trice, and make some different choice in the middle, you may want to follow the best candidates' trice.
# 3.Remember, longer action chain doesn't mean is closer to success, because the process can not rollback, you may refer to the example as "what is  a success". 
# Begin:
# 1.it's your turn to first analyze each candidates. One cnadiates one sentence.
# 2.Then you can tell me who do you think is most close to success, and why.
# 3.Finally you must tell which is the best in one Word. with the following format:
# analyze candidate 1: xxx
# analyze candidate 2: xxx
# ...
# analyze candidate n: xxx
# most close to succeess: xxx
# best candidate: "candidate x" / "tied"
# Begin!

llm = chatGPT()
re = llm.generation("",vote_prompt_relative,model="text-davinci-003")
print(re)


