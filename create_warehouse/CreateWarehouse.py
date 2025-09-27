import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
import pygame

"""
This class is used to easily create any warehouse configuration with given parameters * :
    - Number of blocks
    - Number of aisles
    - Number of locations per aisle
    
For information about how to use the class, please consult the example_create_warehouse.py  
    

Blocks are colour coded with the following encoding rules:

            Colour encoding:
                -White: Free space / Traversable
                -Black: Empty storage location / Untraversable
                -Red: Storage location / Untraversable
                -White: Pick-up location / Traversable
                -Green: Start and end position / Traversable
                
To save the warehouse click the SPACE button. This will create a sub-folder in the specified parent directory.
The subfolder will contain a numerically encoded warehouse following the encoding rules:

            Numerical encoding:
                -0: Free space / Traversable
                -1: Empty storage location / Untraversable
                -2: Storage location / Untraversable
                -3: Pick-up location / Traversable
                -9: Start and end position / Traversable
                






Author: Ziga Breznikar
Mail: ziga.breznikar@student.um.si
Date: 22.09.2025

1. Edit: 27.09.2025

"""



class CreateWarehouse:
    """
    This class is responsible for setting the parent directory in which you want to create different warehouse instances.
    Warehouses are created with the use of GUI, developed with Pygame.
    For encoding explanation, please view draw_warehouse method.

    ----------
    Parameters:
    ----------
        self: CreateWarehouse
            Instance of class used to create and save the warehouses.

        parent_directory: str
            Name of the parent directory in which the warehouses folders will be created.

        warehouse_name str :
            Name of the specific warehouse

        save_to_text: bool
            If set to True, the warehouse will be saved to in the specified parent directory. Defaults to False.

    """


    def __init__(self,parent_directory,warehouse_name,save_to_text=False):
        self.parent_directory = parent_directory
        self.warehouse_name = warehouse_name
        self.save_to_text = save_to_text




    def create_row(self,num_of_aisles):
        """

        Creates a single row of the warehouse block.
        Its main purpose is creating a single row of a block which will be iteratively added to create the entire block.

        ----------
        Parameters:
        ----------

        self: CreateWarehouse
            Instance of class used to create and save the warehouses.

        num_of_aisles: int
            Number of aisles.

        -------
        Returns:
            row: np.ndarray
                A single block of the warehouse.
        -------

        """

        row = np.zeros(3 * num_of_aisles)
        num_of_coloumns = 3 * num_of_aisles
        index = 0
        for i in range(2, num_of_coloumns):
            if index == 2:
                index = 0
            else:
                row[i] = 1
                index = index + 1
        row[0] = 1
        row[num_of_coloumns - 1] = 1
        return row

    def create_block(self,num_of_loc_aisles,num_of_aisles):
        """
        Helper function in create_warehouse_array function.
        Its main purpose is creating a single block of a warehouse which will be iteratively added to create the entire warehouse.

        ----------
        Parameters:
        ----------
        self: CreateWarehouse
            Instance of class used to create and save the warehouses.

        num_of_aisles: int
            Number of aisles.

        num_of_loc_aisles: int
            Number of storage locations per aisle.

        -------
        Returns:
            block: np.ndarray
                A single block of the warehouse.
        -------

        """

        block = np.array([self.create_row(num_of_aisles) for _ in range(num_of_loc_aisles)])
        return block


    def create_warehouse_array(self,num_of_blocks, num_of_aisles, num_of_loc_aisles, depot_location=None):
        """
        Method is a helper function to create an empty warehouse array, used in create_warehouse method.
        Its main purposes is creating a warehouse array, that is used later used for selecting pick-up locations in GUI.

        ----------
        Parameters:
        ----------
        self: CreateWarehouse
            Instance of class used to create and save the warehouses.

        num_of_blocks: int
            Number of storage blocks.

        num_of_aisles: int
            Number of aisles.

        num_of_loc_aisles: int
            Number of storage locations per aisle.

        depot_location: array
            Start and end position of AGV.
            In format of [x,y].
            Defaults to None. Meaning it is placed at [0,1].

        -------
        Returns:
            warehouse_array: np.ndarray
                Warehouse array in an encoded format.
        -------

        """

        try:
            if depot_location is None:
                depot_location = [0,1]
            path_arr = np.zeros(3 * num_of_aisles)
            path_arr[0], path_arr[-1] = float("inf"), float("inf")
            warehouse_array = [path_arr]  # Initialize with the first path row

            for _ in range(num_of_blocks):
                block = self.create_block(num_of_loc_aisles, num_of_aisles)  # Generate a block (2D)
                warehouse_array.append(block)  # Append the block (entire 2D structure)
                warehouse_array.append(path_arr)  # Append another path row after the block

            warehouse_array = np.vstack(warehouse_array)
            warehouse_array[depot_location[0]][depot_location[1]] = 9

            return warehouse_array

        except Exception as e:
            print(f"Invalid input format, consult documentation.\n Error: {e}")

    def add_locations_to_warehouse(self, warehouse_array, locations):
        """
        Used for adding pick-up locations in the warehouse GUI.
        Method places a pick-up location to the right or to the left of the selected storage locations.

        ----------
        Parameters:
        ----------
        self: CreateWarehouse
            Instance of class used to create and save the warehouses.

        warehouse_array: np.ndarray
            Warehouse array in an encoded format.

        locations: array
            Contains positions of storage locations where we want to the AGV to visit.

        ----------
        Returns:
            warehouse_array: np.ndarray
                Warehouse array in an encoded format containing pick-up locations for the AGV to visit.
        ----------

        """

        for loc in locations:
            try:  # try catch for invalid input format
                if warehouse_array[loc[0]][loc[1]] == 2:
                    try:  # Try to put the pickup place on the right
                        if warehouse_array[loc[0]][loc[1] + 1] == 0:
                            warehouse_array[loc[0]][loc[1] + 1] = 3
                    except Exception as e:
                        #print("Cannot put pick up place at the right!")
                        pass
                    try:  # Try to put the pickup place on the left
                        if warehouse_array[loc[0]][loc[1] - 1] == 0:
                            warehouse_array[loc[0]][loc[1] - 1] = 3
                    except Exception as e:
                        #print("Cannot put pick up place at the left!")
                        pass
                else:
                    print("Invalid location, try a different one")
            except Exception as e:
                print(f"Invalid input format, consult documentation. Error: {e}")


        return warehouse_array

    def draw_warehouse(self,warehouse_array,created_directory,root=None):
        """
        Method is used to draw the warehouse. Basic building blocks of the warehouse, are blocks :).
        Each block has two different forms of encoding:
            Colour encoding:
                -White: Free space / Traversable
                -Black: Empty storage location / Untraversable
                -Red: Storage location / Untraversable
                -White: Pick-up location / Traversable
                -Green: Start and end position / Traversable

            Numerical encoding:
                -0: Free space / Traversable
                -1: Empty storage location / Untraversable
                -2: Storage location / Untraversable
                -3: Pick-up location / Traversable
                -9: Start and end position / Traversable

        ----------
        Parameters:
        ----------

        self: CreateWarehouse
            Instance of class used to create and save the warehouses.

        warehouse_array: np.ndarray
            Warehouse array in an encoded format, created with create_warehouse_array method.

        created_directory: str
            Name of the warehouse directory.

        -------
        Returns: None
        -------

        """

        warehouse = warehouse_array.copy()
        pygame.init()

        max_width = 1200
        max_height = 790
        rows, cols = warehouse_array.shape
        CELL_SIZE = min(max_width // cols, max_height // rows)
        width = warehouse_array.shape[1] * CELL_SIZE
        height = warehouse_array.shape[0] * CELL_SIZE
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Warehouse Layout")

        #Path to save
        full_path = os.path.join(self.parent_directory, created_directory)

        # Colors
        WHITE = (255, 255, 255)  # Empty space and paths
        BLACK = (0, 0, 0)  # Storage locations (1)
        RED = (255, 0, 0)  # Terminal locations/ pickups (2)
        GREEN = (0, 255, 0)  # Start/End location (9)

        locations = []

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:  # If users clicks
                    x, y = pygame.mouse.get_pos()  # Dobimo position kjer smo kliknili
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE

                    if warehouse_array[grid_y, grid_x] == 1:  # Ce kliknemo na empty jo setamo na full
                        warehouse_array[grid_y, grid_x] = 2
                        locations.append([grid_y, grid_x])
                    elif warehouse_array[grid_y, grid_x] == 2:
                        warehouse_array[grid_y, grid_x] = 1
                        try:
                            locations.remove([grid_y, grid_x])
                        except ValueError:
                            pass
                elif event.type == pygame.KEYDOWN:  # Confirmation, user klikne enter da confirma warehouse
                    if event.key == pygame.K_SPACE:
                        warehouse = self.add_locations_to_warehouse(warehouse, locations)
                        if self.save_to_text:
                            try: # Ustavari mapo
                                os.makedirs(full_path, exist_ok=True)
                                print(f"Directory '{full_path}' created successfully")
                            except Exception as e:
                                print(f"Failed to create directory '{full_path}' Error: {e}")

                            try: # Tries to save warehouse in array format
                                warehouse = np.where(np.isinf(warehouse), -1, warehouse).astype(int)
                                np.savetxt( # Saves the warehouse in array format in parent_directory/created_directory to file created_directory.txt
                                    self.parent_directory+"/" + created_directory + "/" + created_directory + ".txt",
                                    warehouse, fmt="%d", delimiter=" ")
                                print("Sucessfully saved warehouse.")
                            except Exception as e:
                                print(f"Could not save warehouse. Error: {e}")


                            try: # Tries to save the image of created warehouse
                                screen_shot_path = os.path.join(full_path, f"{created_directory}.png")
                                pygame.image.save(screen, screen_shot_path)
                                print(f"Image saved to '{screen_shot_path}'")
                            except Exception as e:
                                print(f"Could not save screenshot. Error: {e}")
                        running = False

                        if root is not None:
                            root.destroy()

                    # print(warehouse_array) # Uncomment to print array version of warehouse

            screen.fill(WHITE)  # Default background
            for y in range(warehouse_array.shape[0]):
                for x in range(warehouse_array.shape[1]):
                    if warehouse_array[y, x] == 1:
                        color = BLACK  # Storage location
                    elif warehouse_array[y, x] == 2:  # Terminal location
                        color = RED
                    elif warehouse_array[y, x] == 9:  # Depot location
                        color = GREEN
                    else:  # Vse ostalo
                        color = WHITE
                    pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, (200, 200, 200), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

            # screen.blit(instruction_text, (10, 10)) # Kam damo text

            pygame.display.flip()

        # print(warehouse)

        pygame.quit()
        return None

    def draw_empty_warehouse(self,warehouse_array,root=None):
        """
        Method is used to draw the empty warehouse, so the user can visually confirm it is correct.
        Each block has two different forms of encoding:
            Colour encoding:


            Numerical encoding:

        ----------
        Parameters:
        ----------

        self: CreateWarehouse
            Instance of class used to create and save the warehouses.

        warehouse_array: np.ndarray
            Warehouse array in an encoded format, created with create_warehouse_array method.

        -------
        Returns: None
        -------
        """

        warehouse = warehouse_array.copy()
        pygame.init()

        max_width = 1200
        max_height = 790
        rows, cols = warehouse_array.shape
        CELL_SIZE = min(max_width // cols, max_height // rows)
        width = warehouse_array.shape[1] * CELL_SIZE
        height = warehouse_array.shape[0] * CELL_SIZE
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Empty Warehouse Layout")

        # Path for the saving of empty warehouse
        helper = f"{self.warehouse_name}_empty"
        full_path = os.path.join(self.parent_directory, self.warehouse_name,helper)
        scenario_path = os.path.join(self.parent_directory, self.warehouse_name,"scenarios")


        # Colors
        WHITE = (255, 255, 255)  # Empty space and paths
        BLACK = (0, 0, 0)  # Storage locations (1)
        GREEN = (0, 255, 0)  # Start/End location (9)

        running = True

        while running:
            for event in pygame.event.get(): # For quiting
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:  # If user clicks space bar it closes or closes and saves
                    if event.key == pygame.K_SPACE:
                        if self.save_to_text:

                            try: # Try to create directory for empty warehouse and scenario directory
                                os.makedirs(full_path, exist_ok=False)
                                os.makedirs(scenario_path, exist_ok=False)
                                print(f"Directory '{full_path}' created successfully")
                            except Exception as e:
                                print(f"Failed to create directory '{full_path}' Error: {e}")

                            try: # Tries to save warehouse in array format
                                warehouse = np.where(np.isinf(warehouse), -1, warehouse).astype(int)
                                np.savetxt(
                                    os.path.join(full_path,self.warehouse_name+"_empty.txt"),
                                    warehouse, fmt="%d", delimiter=" ")
                                print("Successfully saved warehouse in array format.")
                            except Exception as e:
                                print(f"Could not save warehouse. Error: {e}")


                            try: # Try to save png image of created warehouse
                                screen_shot_path = os.path.join(full_path, f"{self.warehouse_name}_empty.png")
                                pygame.image.save(screen, screen_shot_path)
                                print(f"Successfully saved empty warehouse in png format")
                            except Exception as e:
                                print(f"Could not save screenshot. Error: {e}")
                        running = False

                        if root is not None:
                            root.destroy()



            screen.fill(WHITE)  # Default background
            for y in range(warehouse_array.shape[0]):
                for x in range(warehouse_array.shape[1]):
                    if warehouse_array[y, x] == 1:
                        color = BLACK  # Storage location
                    elif warehouse_array[y, x] == 9:  # Depot location
                        color = GREEN
                    else:  # Vse ostalo
                        color = WHITE
                    pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, (200, 200, 200), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

            pygame.display.flip()

        pygame.quit()
        return None



    def create_empty_warehouse(self):
        """
        Method is used for creating an empty warehouse with desired parameters (see *)
        Method will create an empty warehouse (.txt file and image) in warehouse_name/warehouse_name_empty directory.
        The .txt file which includes the empty warehouse layout will be used as a blueprint when setting desired pick-up
        locations.

        Method will also create an empty folder called scenarios in which different pickup scenarios will be created,
        by the create_pickup_scenario method.

        ----------
        Parameters:
        ----------
            self: CreateWarehouse

        warehouse_name: str
            Name of the warehouse, will create a subdirectory with the given name, it will include the blueprint
            of the empty warehouse in matrix form which will be used for setting desired pick-up locations.

        -------
        Returns: None
        -------
        """

        root = tk.Tk()
        root.title("Empty Warehouse Creator")

        #---------------------------------------Labels and Entry fields------------------------------------------------#

        # Number of blocks:
        tk.Label(root, text="Number of Blocks:").grid(row=0, column=0, padx=5, pady=5)
        blocks_entry = tk.Entry(root)
        blocks_entry.grid(row=0, column=1, padx=5, pady=5)

        # Number of aisles:
        tk.Label(root, text="Number of Aisles:").grid(row=1, column=0, padx=5, pady=5)
        aisles_entry = tk.Entry(root)
        aisles_entry.grid(row=1, column=1, padx=5, pady=5)

        # Locations per aisle
        tk.Label(root, text="Locations per Aisle:").grid(row=2, column=0, padx=5, pady=5)
        loc_aisles_entry = tk.Entry(root)
        loc_aisles_entry.grid(row=2, column=1, padx=5, pady=5)

        #----------------------------------------Button to plot warehouse----------------------------------------------#

        def on_plot():
            try:
                num_of_blocks = int(blocks_entry.get())
                num_of_aisles = int(aisles_entry.get())
                num_of_loc_aisles = int(loc_aisles_entry.get())
                if num_of_blocks <= 0 or num_of_aisles <= 0 or num_of_loc_aisles <= 0:
                    raise ValueError("All inputs must be positive integers.")
                warehouse_array = self.create_warehouse_array(num_of_blocks, num_of_aisles, num_of_loc_aisles)
                self.draw_empty_warehouse(warehouse_array,root) # This has to be modified for only empty warehouse
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e) if str(e) else "Please enter valid integers.")

        #----------------------------------------Button to reset warehouse---------------------------------------------#
        def on_reset():  # Button to reset warehouse
            blocks_entry.delete(0, tk.END)
            aisles_entry.delete(0, tk.END)
            loc_aisles_entry.delete(0, tk.END)

        tk.Button(root, text="Plot Warehouse", command=on_plot).grid(row=3, column=0, columnspan=2, pady=10)
        tk.Button(root, text="Reset Warehouse", command=on_reset).grid(row=4, column=0, columnspan=2, pady=10)
        tk.Label(root, text="1. Select warehouse plotting parameters.\n "
                        "2. Add locations in warehouse, click on the black boxes.\n"
                        "3. Confirm selection press the SPACE key",font=("Default",10)).grid(row=5, column=0,
                                                                                             columnspan=2, pady=10)

        root.mainloop()

        return None


    def create_pickup_scenario(self,pickup_scenario,root=None):
        """
        Method is used for creating pick-up scenarios on a warehouse configuration.
        It will use the warehouse_name_empty.txt and change it to match the desired pick-up scenario.
        The pickup scenario will be saved in parent_directory/warehouse_name/name_of_pickup_scenario.

        -----------
        Parameters:
        -----------

        self: CreateWarehouse

        pickup_scenario: str
            Name of pick-up scenario.

        --------
        Returns: None
        --------

        """

        full_path_save = os.path.join(self.parent_directory,self.warehouse_name,"scenarios",pickup_scenario)
        full_path_import = os.path.join(self.parent_directory,self.warehouse_name,
                                        f"{self.warehouse_name}_empty",f"{self.warehouse_name}_empty.txt")


        warehouse_array = np.loadtxt(full_path_import, dtype=int) # Load empty warehouse

        pygame.init()

        max_width = 1200
        max_height = 790
        rows, cols = warehouse_array.shape
        CELL_SIZE = min(max_width // cols, max_height // rows)
        width = warehouse_array.shape[1] * CELL_SIZE
        height = warehouse_array.shape[0] * CELL_SIZE
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Warehouse: {self.warehouse_name} ")

        # Colors
        WHITE = (255, 255, 255)  # Empty space and paths
        BLACK = (0, 0, 0)  # Storage locations (1)
        RED = (255, 0, 0)  # Terminal locations/ pickups (2)
        GREEN = (0, 255, 0)  # Start/End location (9)

        locations = []

        running = True
        while running:

            # Block for quiting
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Block for selecting/deselecting pick-up locations
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE

                    # If storage empty set, to full
                    if warehouse_array[grid_y, grid_x] == 1:
                        warehouse_array[grid_y, grid_x] = 2
                        locations.append([grid_y, grid_x])


                    # If storage full set, to empty
                    elif warehouse_array[grid_y, grid_x] == 2:
                        warehouse_array[grid_y, grid_x] = 1
                        try:
                            locations.remove([grid_y, grid_x])
                        except ValueError:
                            pass

                # Block for saving
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print(locations)
                        warehouse_array = self.add_locations_to_warehouse(warehouse_array, locations)

                        if self.save_to_text:
                            try: # Try to create directory for pick-up scenario
                                os.makedirs(full_path_save, exist_ok=False)
                                print(f"Successfully created pick up scenario directory: {full_path_save}")
                            except Exception as e:
                                print(f"Failed to create directory. Error: {e}")

                            try: # Try to save pick-up scenario in array format
                                warehouse_array = np.where(np.isinf(warehouse_array), -1, warehouse_array).astype(int)
                                np.savetxt(os.path.join(full_path_save,f"{pickup_scenario}.txt"),warehouse_array,
                                           fmt="%d", delimiter=" ")
                                print(f"Successfully saved pick up scenario in array format!")
                            except Exception as e:
                                print(f"Failed to save warehouse in array format. Error: {e}")

                            try: # Try to save png image of pick-up scenario
                                pygame.image.save(screen, os.path.join(full_path_save,f"{pickup_scenario}.png"))
                                print(f"Successfully saved picked up scenario in png format!")
                            except Exception as e:
                                print(f"Failed to save warehouse in png format. Error: {e}")
                        running = False

                        if root is not None:
                            root.destroy()

            screen.fill(WHITE)  # Default background
            for y in range(warehouse_array.shape[0]):
                for x in range(warehouse_array.shape[1]):
                    if warehouse_array[y, x] == 1:
                        color = BLACK  # Storage location
                    elif warehouse_array[y, x] == 2:  # Terminal location
                        color = RED
                    elif warehouse_array[y, x] == 9:  # Depot location
                        color = GREEN
                    else:  # Vse ostalo
                        color = WHITE
                    pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, (200, 200, 200), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

            pygame.display.flip()

        pygame.quit()
        return None







