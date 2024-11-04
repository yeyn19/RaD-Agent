from typing import List, Set
import os
import json
import json
import os
from typing import Any, Dict, List
import re
from typing import Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import ZeroShotAgent
from langchain.schema import AgentFinish
from langchain.agents import AgentExecutor
import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish
from typing import NamedTuple

NAME2URL = {
    "datasette": "https://datasette.io/",
    "klarna": "https://www.klarna.com/",
    'chemical-prop':  "http://127.0.0.1:8079/tools/chemical-prop/",
    'douban-film':  "http://127.0.0.1:8079/tools/douban-film/",
    'weather':  "http://127.0.0.1:8079/tools/weather/",
    'wikipedia':  "http://127.0.0.1:8079/tools/wikipedia/",
    'wolframalpha':  "http://127.0.0.1:8079/tools/wolframalpha/",
    'klarna':  "https://www.klarna.com/",
    "meta_analysis": "http://127.0.0.1:8079/tools/meta_analysis/"
}

def read_queries(file_path: str) -> Set[str]:
    query_set = set()
    with open(file_path, "r") as f:
        queries = f.readlines()
        for query in queries:
            query_set.add(query)
    return query_set

def prepare_queries(query_path: str, answer_path: str) -> List[str]:
    queries = read_queries(query_path)
    # check already generated queries
    if os.path.exists(answer_path):
        pass_queries = [json.loads(line)["query"] for line in open(answer_path, "r")]
        for query in pass_queries:
            if query in queries:
                queries.remove(query)
    return list(queries)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fix(input_list: list, length: int) -> list:
    while len(input_list) < length:
        input_list.append("")
    return input_list

class MyZeroShotAgent(ZeroShotAgent):
    @property   
    def return_values(self) -> List[str]:
        """Return values of the agent."""
        return ["output", "log"]

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return ""

class MyAgentExecutor(AgentExecutor):

    def _return(self, output: AgentFinish, intermediate_steps: list) -> Dict[str, Any]:
        self.callback_manager.on_agent_finish(
            output, color="green", verbose=self.verbose
        )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        final_output["log"] = output.log
        return final_output

    async def _areturn(
        self, output: AgentFinish, intermediate_steps: list
    ) -> Dict[str, Any]:
        if self.callback_manager.is_async:
            await self.callback_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        else:
            self.callback_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        final_output["log"] = output.log
        return final_output


class MyMRKLOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        FINAL_ANSWER_ACTION = "Final Answer:"
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        # \s matches against tab/newline/whitespace
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            return AgentFinish(
                {"output": text}, text
            )
            # raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), text)


class AgentReactResponse(NamedTuple):
    """Agent's return value."""
    thought: list
    action: list
    action_input: list


class LogParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        thought_regex = r"Thought: (.*)\n"
        action_regex = r"Action: (.*)\n"
        input_regex = r"Action Input: (.*)\n"
        text += "\n"
        thought = re.findall(thought_regex, text)
        action = re.findall(action_regex, text)
        action_input = re.findall(input_regex, text)
        
        max_chain = max(len(thought), len(action), len(action_input))
        thought = fix(thought, max_chain)
        action = fix(action, max_chain)
        action_input = fix(action_input, max_chain)
        
        return AgentReactResponse(
            thought, action, action_input
        )
        

