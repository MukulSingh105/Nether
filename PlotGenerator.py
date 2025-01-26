from abc import ABC
from typing import *
from Blocks.Plot import Plot
from Redstone.Task import Task


class PlotGenerator(ABC):
    ## how to define task
    def __init__(self, task: Task):
        self.task = task
        llm_client = None
    
    """
    Generates the prompt to send to LLM for plot generation.
    """
    def generate_prompt(self) -> str:
        template = open("Maps/PlotGeneration.md", "r").read()
        return template.replace("{{Summary}}", self.task.plot_string)

    """
    Parses the LLM response (from generate_response) into Plot class from `Blocks/Plot`
    """
    def parse_plot(self) -> Plot:
        pass
    
    """
    LLM call to gneerate plot
    """
    def generate_plot_response(self) -> str:
        prompt = self.get_prompt()
        response = self.llm_client.send_request(prompt)
        plot = self.parse_plot()