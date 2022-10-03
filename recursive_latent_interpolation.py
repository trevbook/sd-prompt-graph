import os
os.chdir("stable-diffusion")
from pathlib import Path
import json
from prompt_graph_utils import prompt_graph_back_and_forth_recursion

input_folder = Path(input("Enter the path to the folder of images you're trying to smooth generation of: "))

# Open the configuration folder
with open(input_folder/Path("config.json"), "r") as json_file:
    config = json.load(json_file)

# Run the back and forth interpolation
prompt_graph_back_and_forth_recursion(input_folder, loops=3,
                                      pct_change_threshold=0.05, early_stop=3,
                                      seed=config["seed"])