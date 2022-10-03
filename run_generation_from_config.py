# Import the os module, and then change the working directory to the stable-diffusion/ folder
import os
os.chdir("stable-diffusion/")

# Now, import the other necessary modules
from prompt_graph_utils import launch_config_generation
from pathlib import Path

# Now, load in the path from the user input
config_file_path = Path(input("Enter the path to the configuration file you'd like to generate: "))

# Run the generation
output_file_path = launch_config_generation(config_file_path)

# Print some information
print(f"\n\nGeneration has finished! Your generation is located at {output_file_path}")

