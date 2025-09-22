from CreateWarehouse import CreateWarehouse

"""
This script serves as example of usage.

If you want to create a warehouse using the Class: CreateWarehouse you just make an instance
of the class.

You must specify a parent directory (ex. example_parent_dir) in which the warehouses will be saved. A parent directory can
contain multiple folders, each corresponding to a different warehouse.

You must specify whether or not to save the created warehouse. This is done by setting save_to_text to True.

After creating an instance call the: .create_warehouse instance and specify the warehouse name (ex. example_warehouse_1)

When selecting all the storage locations AGV should visit place SPACE_BAR to save.


"""
# Creates an instance, setting the parent directory to: example_parent_dir . This will save all the warehouses created from this instance into mentioned directory.
warehouse_example = CreateWarehouse(parent_directory="example_parent_dir",save_to_text=True)

# This will open the GUI, which will guide you through warehouse creation
warehouse_example.create_warehouse("example_warehouse_1")