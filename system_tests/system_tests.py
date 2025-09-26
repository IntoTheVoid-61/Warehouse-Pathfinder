from CreateWarehouse import CreateWarehouse
from FindPath import FindPath
import yaml


parent_directory = r"C:\Users\ACER\PycharmProjects\Machine_Learning\Komisioniranje\Class_implementation\system_tests"
warehouse_example = CreateWarehouse(parent_directory=parent_directory,warehouse_name="warehouse_tests",save_to_text=True)
#------------------------------------------------Creating empty warehouse----------------------------------------------#
#warehouse_example.create_empty_warehouse()

#------------------------------------------------Creating pick-up scenarios--------------------------------------------#
#warehouse_example.create_pickup_scenario(pickup_scenario="system_test_pickup_scenario_2")

#------------------------------------------------Read pick-up scenario-------------------------------------------------#
with open("system_test_config.yaml", "r") as f:
    config = yaml.safe_load(f)

params = config["find_path"]
find_path = FindPath(**params)

find_path.import_pickup_scenario("system_test_pickup_scenario_1")
