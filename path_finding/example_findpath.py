import yaml
from FindPath import FindPath

"""
This script serves as example of usage.

It will run the algorithm on a selected pickup scenario, created with CreateWarehouse method.
The results will be saved inside the: parent_dir/warehouse_name/scenarios/scenario_name/results/results.json 

results.json stores the calculated path length and the steps needed to take, to achieve the desired path length.

"""

# Load config file
with open("config/fast_solution.yaml", "r") as f:
    config = yaml.safe_load(f)

# store params
params = config["find_path"]
# create an instance of FindPath class
find_path = FindPath(**params)

# Run the pathfinding algorithm
find_path.find_path("example_pickup_scenario")

