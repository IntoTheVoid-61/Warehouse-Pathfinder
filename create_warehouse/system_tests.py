from CreateWarehouse import CreateWarehouse

warehouse = CreateWarehouse(parent_directory="system_tests",warehouse_name="warehouse_test",save_to_text=True)

#warehouse.create_empty_warehouse()
warehouse.create_pickup_scenario(pickup_scenario="scenario_test_4")
