from CreateWarehouse import CreateWarehouse

"""
This script serves as example of usage.

If you want to create a warehouse using the Class: CreateWarehouse you just make an instance
of the class.

Creating an CreateWarehouse instance:
    
    CreateWarehouse(parent_directory,warehouse_name,save_to_text)
    
        parent_directory str :
            You must specify a parent directory (ex. example_parent_dir) in which the warehouses will be saved. A parent directory can
            contain multiple warehouse folders. This can be the same for multiple instances
            
        warehouse_name str :        
            Name of the specific warehouse
                         
        save_to_text bool :
            True if you want to save the warehouse.
            Default: False


Creating empty warehouse:

    .create_empty_warehouse()
       
    This will open the GUI, which will guide you through warehouse creation of an empty_warehouse.
    To save/close press SPACE_BAR.
    
    It will create an warehouse_name_empty subfolder inside the warehouse_name folder.
    It contains:
        warehouse_name_empty.txt which is the array encoding of an empty warehouse.
        warehouse_name_empty.png which is a visualization of an empty warehouse. 
        
    Method will also create an empty folder inside /parent_directory/warehouse_name 
    called scenarios in which different pickup scenarios will be created.
    
Creating pick-up scenarios:

    create_pickup_scenario(pick_up_scenario)
    
    This will open the GUI, where you select desired pick-up locations in the empty warehouse.
    To save/close press SPACE_BAR.
    
    It creates a subfolder in scenarios folder.
    It contains:
        pick_up_scenario.txt which is the array encoding of a pick-up scenario.
        pick_up_scenario.png which is a visualization of a pick-up scenario.
        
        pick_up_scenario str :
            Name of pickup scenario
            

                       
It is recommended that a single instance of class CreateWarehouse is used for one empty warehouse configuration.


Once the empty warehouse is made you can create multiple pick-up scenarios using create_pickup_scenario method.
    Each new pick-up scenario will be saved with the given name: pickup_scenario in the scenarios folder,
    inside of warehouse_name folder.

"""
# Creates an instance, setting the parent directory to: example_parent_dir . This will save all the warehouses created from this instance into mentioned directory.
# warehouse_name corresponds to specific warehouse configuration
warehouse_example = CreateWarehouse(parent_directory="../example_parent_dir",warehouse_name="example_warehouse_1",save_to_text=True)

# This will open the GUI, which will guide you through warehouse creation of an empty_warehouse, when finished enter SPACE_BAR to save.
warehouse_example.create_empty_warehouse()

# This will open GUI, where you can specify a pick-up scenario, by clicking on empty storage locations (black). To save press SPACE_BAR.
warehouse_example.create_pickup_scenario(pickup_scenario="example_pickup_scenario")

