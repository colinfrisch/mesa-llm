import os

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_space_component,
)

from examples.negotiation.agents import BuyerAgent, SellerAgent
from examples.negotiation.model import NegotiationModel
from mesa_llm.reasoning.rewoo import ReWOOReasoning

load_dotenv()


def model_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 25,
    }

    if isinstance(agent, SellerAgent):
        portrayal["color"] = "tab:red"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2
    elif isinstance(agent, BuyerAgent):
        portrayal["color"] = "tab:blue"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 1

    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_buyers": 1,
    "width": 4,
    "height": 4,
    "api_key": os.getenv("GEMINI_API_KEY"),
    "reasoning": ReWOOReasoning,
    "llm_model": "gemini/gemini-2.0-flash",
    "vision": 5,
}


model = NegotiationModel(
    initial_buyers=model_params["initial_buyers"],
    width=model_params["width"],
    height=model_params["height"],
    api_key=model_params["api_key"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"]["value"],
)

page = SolaraViz(
    model,
    components=[make_space_component(model_portrayal)],
    model_params=model_params,
    name="Negotiation",
)

page  # noqa

"""run with:
conda activate mesa-llm && solara run examples/negotiation/app.py
"""
