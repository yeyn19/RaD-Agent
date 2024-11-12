# 📖 Overview

<img src="./assets/method.png">

With remarkable advancements, large language models (LLMs) have attracted significant efforts to develop LLM-based agents capable of executing intricate multi-step decision-making tasks. Existing approaches predominantly build upon the external performance measure to guide the decision-making process but the reliance on the external performance measure as prior is problematic in real-world scenarios, where such prior may be unavailable, flawed, or even erroneous. For genuine autonomous decision-making for LLM-based agents, it is imperative to develop rationality from their posterior experiences to judge the utility of each decision independently. 

In this work, we propose RaDAgent (**Ra**tional **D**ecision-Making **Agent**), which fosters the development of its rationality through an iterative framework involving Experience Exploration and Utility Learning. Within this framework, Elo-based Utility Learning is devised to assign Elo scores to individual decision steps to judge their utilities via pairwise comparisons. Consequently, these Elo scores guide the decision-making process to derive optimal outcomes. 

Experimental results on the Game of 24, WebShop, ToolBench and RestBench datasets demonstrate RaDAgent’s superiority over baselines, achieving about 7.8% improvement on average. Besides, RaDAgent also can reduce costs (ChatGPT API calls), highlighting its effectiveness and efficiency.

Our paper is released [here](https://arxiv.org/abs/2308.12519).

# ⚙️ Environment Setup

Here is the guideline of how to run RaDAgent method in different downstream-tasks.

You need to first clone the repo and install the project dependency:

```bash
python -m pip install -e ./
```

Then, before you start, you need to put your OpenAI API Keys in someplace of the Repo `ets_utils.py`. 

# 🚀 Running

To run on the downstream tasks we report in the paper, use the following guides.

## 🛒 Webshop

### Start Server

run the following commands:

```bash
cd run_webshop
python env_server.py
```

The IP and port can be set in `env_server.py`.

```
host = "localhost" # your host ip
port = 12348 # your host port
```

### Run Benchmark

```bash
cd run_webshop
bash run_task.sh
```

You can set the connection ip and port in `run_webshop\webshopping_env_online.py`.

### Evaluation

```bash
cd run_webshop
bash eval.sh
```



## 🌐 RestGPT

Configure the REST API environment (TMDB, Spotify) by following the instructions in [RestGPT](https://github.com/Yifan-Song793/RestGPT).

### Start Server

run the following commands:

```bash
cd run_restbench
python env_server.py
```

The IP and port can be set in `env_server.py`.

```
host = "localhost" # your host ip
port = 12348 # your host port
```

### Run Benchmark

```bash
cd run_restbench
bash run_task.sh
```

You can set the connection ip and port in `run_restbench\restbench_env.py`.

### Evaluation

```bash
cd run_restbench
python restbench_eval.py
```



## 🔢 Game of 24

#### LLM-Based Baseline

We follow the setting of [ToT](https://arxiv.org/abs/2305.10601), which uses a `./Downstream_tasks/24.csv` and test the last 100 problems, which is the hardest in the dataset. Then, you can run the experiment:

```
python ./test_codes/test_24.py
```

You need to specify some arguments like input or output dir, process_num, in that python file. This code is not only for Elo-tree-search, you can also specify DFS, DFSDT, BFS method in the `--method` part.

#### MCTS baseline

Also, we implement an MCTS baseline, based on the traditional UCT tree-search method, it has different versions, the difference in `get_reward` function, you can specify it. To run the baseline, use the follow command:

```
python ./test_codes/MCTS.py
```

> MCTS can also finds a result of each case, but it uses 100 times more simulations than ETS. We give a compare in our paper.

## 🛠 ToolBench

To test on Toolbench, it is a little complex. You need to 

1. request a ToolServer `toolbenchkey` following the guide [here](https://github.com/OpenBMB/ToolBench), also you can build it [locally](https://drive.google.com/file/u/0/d/1JdbHkL2D8as1docfHyfLWhrhlSP9rZhf/view?usp=sharing&pli=1) after getting a toolbench key, you need to specify the `toolbench_key` in `ets_utils.py`
2. add denpendency: We use an early version of toolbench, so you need to unzip the `assets.zip` to unzip the denpendency.

After setting up toolbench env, you can run the commands: 

```bash
python answer_generation.py
```

In our experiment, we use this split```./assetstoolbench_test_data_0925/test_query_ids```.  You can try different methods in `--method` part, like DFS, BFS, ETS. The explanation of the hyperparameters are described in the following part.

> Because gpt-3.5-turbo-0613 can not be Requested by OpenAI, and the main experiement is performed in 2023.07, Many Rapid-API server is not exists today, the score may not re-implemented today. But we have hold the original ToolBench Test result of our main experiment, If you have any problems reimplementing ToolBench experiment,  you can connect yeyn2001@gmail.com

# 📊 Experimental Results

| Model     | Game of 24 | WebShop | ToolBench |
|-----------|------------|---------|-----------|
| CoT       | 6.00       | 56.23   | 16.60     |
| CoT@3     | 7.00       | 56.45   | 31.20     |
| Reflexion | 7.00       | 57.21   | 26.60     |
| ToT-BFS   | 11.00      | 50.20   | 38.00     |
| ToT-DFS   | 14.00      | 55.60   | 45.58     |
| DFSDT     | 29.00      | 57.25   | 50.20     |
| **RaDAgent** | **43.00** | **59.36** | **61.92** |

| Model               | TMDB  | Spotify |
|---------------------|-------|---------|
| Offline             | 33.0  | 36.4    |
| DEPS                | 43.0  | 43.8    |
| ReAct               | 57.0  | 49.1    |
| Reflexion           | 59.0  | 61.4    |
| RestGPT             | 79.0  | 74.5    |
| RestGPT(ChatGPT)    | 65.0  | 72.3    |
| **RaDAgent**        | **84.0** | **80.7** |


# 🔎 Hyperparameter Explanation

Here, we just parse your method name, and specify it into some format ETS method such as `ETS_all-100_annealing_k50_sqrt_s100_f1_t173.72_p0.9_c15_m3_rn1_rg3`. we mainly split the name by "_", the logic is at `./test_codes/test_24.py`, line 340-380

For ETS: 

- ETS means Elo-base Tree Search, the main method of RaD-Agent
- K50, means how to update the elo score in the equation. bigger k means to update elo score fastly. We set this to 50 in all the experiment
- s100, means we will give 100 **S**imulations of one task. (In the main experiment, we just set it 30)
- f1: whether we will generate many candidates in forward pass, This equals to 1 in `DFSDT`
- t173.72. In `Elo Algorithm`, the temperature is set to 173.72

- P0.9, the default probability to explore a new child. bigger P is more like to perform exploration. We set it to 0.5 in main experiment.
- c15: the max number of one node in the tree. In this example, If one node's child count >= 15, we will never generate new nodes for it.
- m3: perform elo backward ever 3 forward. Larger **m** will save API Cost, but the Elo approxmation may be biased. We use 3 in our experiment
- rn1, rg3: When performing a championship, we first let all the newer nodes to complete **rn** times, and randomly select **rg** times global matching

For DFS:

- `woFilter` means to not perform pair-wise comparsion in the forward process. 
  - In the original ToT DFS, this use `Filter` method, which may add many API request.
  - The `DFSDT` methods means to set the `woFilter`
- `w` means the width of the tree. The max child count of one node.

For BFS: BFS follows a process similar to Beam-search

- `w` means the beam-pool size, 
- `e` means all the candidates in the pool have `e` child. So the total cost nears `depth*w*e`  

For CoT:

- `CoT@n` means to try n times. Early-stopping when findding a result

For Reflexion:

- `Reflexion@n` means to try n times. Early-stopping when findding a result

You can read through `./LLM/ETS.py` to see the implemenation of the Elo-based tree search Algorithms

# 📄 Citation
Feel free to cite us if you like our work.
```bibtex
@article{ye2023rational,
  title={Rational Decision-Making Agent with Internalized Utility Judgment},
  author={Ye, Yining and Cong, Xin and Tian, Shizuo and Qin, Yujia and Liu, Chong and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2308.12519},
  year={2023}
}

```

# ⚖️ License
Our project's source code is licensed under the Apache 2.0 License. This license permits the use, modification, and distribution of the code, subject to certain conditions outlined in the Apache 2.0 License.

# 📬 Contact
If you have any questions, please feel free to contact me at `yeyn2001@gmail.com`.