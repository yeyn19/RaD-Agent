# Overview

This is the official implementation of the paper [Rational Decision-Making Agent with Internalized Utility Judgment](https://arxiv.org/abs/2308.12519)

<img src="./images/ets.png">

With remarkable advancements, large language models~(LLMs) have attracted significant efforts to develop LLM-based agents capable of executing intricate multi-step decision-making tasks. Existing approaches predominantly build upon the external performance measure to guide the decision-making process but the reliance on the external performance measure as prior is problematic in real-world scenarios, where such prior may be unavailable, flawed, or even erroneous.For genuine autonomous decision-making for LLM-based agents, it is imperative to develop rationality from their posterior experiences to judge the utility of each decision independently. For genuine autonomous decision-making for LLM-based agents, it is imperative to develop rationality from their posterior experiences to judge the utility of each decision independently.

In this work, we propose **Ra**tional **D**ecision-Making **Agent**(RaD-Agent), which fosters the development of its rationality through an iterative framework involving experience Exploration and Utility Learning. Within this framework, a LLM-based rating system is devised to assign Elo scores to individual decision steps to judge their utilities via pairwise comparisons.  Consequently, these Elo scores guide the decision-making process to derive optimal outcomes.

Experimental results on the Game of 24, WebShop, and ToolBench dataset demonstrate RaD-Agent's superiority over baselines, achieving about 9.3% improvement on average. Besides, RaD-Agent  also can reduce costs (ChatGPT API calls), highlighting its effectiveness and efficiency. 



# Running

Here is the guideline of how to run ETS method in different downstream-tasks



## 1. Webshop

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

## 2. RestGPT

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



## 3. Game of 24

