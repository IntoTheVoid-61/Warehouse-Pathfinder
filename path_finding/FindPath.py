import os
import numpy as np
from collections import deque
import random
import math
import json



"""
This is a Genetic Algorithm based framework for warehouse navigation,
modelled as a Travelling Salesman Problem (TSP) variant for Automated Guided Vehicles (AGVs).
The warehouse layout is represented as a graph, where pick-up locations serve as terminal nodes. 
A distance matrix, compute-ed via Breadth-First Search (BFS) enables efficient route evaluation. 
To promote diversity in the initial population, a Hamming distance-based vectorized initialization strategy is employed,
ensuring that the chromosomes are maximally distinct. 
The GA balances exploration and exploitation by dynamically adjusting the evolutionary parameters. 
Early generations emphasize diversity, while later ones focus on solution refinement, 
improving convergence and avoiding premature stagnation.

GA based solutions perform significantly better when potential solutions (chromosomes) are maximally distinct
and perform poorly when they are similar or equal. Therefore a Hamming distance is used to ensure diversity in the
initial population. 

This implementation is a modified version of the algorithm presented in <add_doi>.
It has been optimized for faster computation, which is critical for real-world AGV applications.
As a result, some features were simplified or removed.

Several YAML configuration files are provided, allowing you to tailor the algorithm to your objectives. 
    - <add this>    

For information about how to use the class, please consult the example.py

P.S.
    Since there are no analytical solutions to the travelling salesman problem I use the term "optimal path"
    as equivalent with the term "the best path that the algorithm found".

Author: Ziga Breznikar
Mail: ziga.breznikar@student.um.si
Date: 22.09.2025

"""


class FindPath:
    """
    This class implements the FindPath algorithm. An instance of this class can operate on any warehouse inside the
    parent_directory.

    Parameters:
    ----------
    self: FindPath
        A FindPath instance that contains the GA framework used to compute optimal path.

    parent_directory: str
         Name of the parent directory in which the warehouses are stored.

    warehouse_name: str
        Name of the specific warehouse

    population_size: int
        Integer value representing the size of the population. In other words this represents the number of
        potential solutions (chromosomes).
        An increase of this will correspond to an increase in computational time.
        If you want quick solutions lower it.
        default: 100

    num_generations: int
        Integer value representing the number of total generations to evolve.
        An increase of this will correspond to an increase in computational time.
        If you want quick solutions lower it.
        default: 1000

    hamming_attempts: int
        Number of hamming attempts to create a new distinct chromosome, if it does not satisfy the criteria in
        hamming_attempts then we lower the criteria by one and repeat the process.

    mutation_rate_min_start: float
        A float value representing the initial lower boundary (a) mutation rate of the GA framework.
        Mutation probability is dynamic and stochastic, sampled from a uniform distribution U(a,b).
        default: 0.1

    mutation_rate_min_end: float
        A float value representing the final lower boundary (a) mutation rate of the GA framework.
        Mutation probability is dynamic and stochastic, sampled from a uniform distribution U(a,b).
        default: 0.01

    mutation_rate_max_start: float
        A float value representing the initial upper boundary (b) mutation rate of the GA framework.
        Mutation probability is dynamic and stochastic, sampled from a uniform distribution U(a,b).
        default: 0.3

    mutation_rate_max_end: float
        A float value representing the final upper boundary (b) mutation rate of the GA framework.
        Mutation probability is dynamic and stochastic, sampled from a uniform distribution U(a,b).
        default: 0.05

    crossover_rate: float
        A float value representing the crossover rate of the GA framework.
        Each selected chromosome has a probability of crossover_rate for crossover,
        and probability of 1-crossover_rate for reproduction.
        default: 0.8

    tournament_size_min: int
        An integer value representing the minimum tournament size of the GA framework.
        Tournament size has linear properties and is proportional to ratio of current generation,
        to number of total generations.
        Value need to be adjusted intelligently if changing population size.
        default: 2

    tournament_size_max: int
        An integer value representing the maximum tournament size of the GA framework.
        Tournament size has linear properties and is proportional to ratio of current generation,
        to number of total generations.
        Value need to be adjusted intelligently if changing population size.
        default: 20

    min_elitism_size: float
        A float value representing the fraction of minimum elitism size of the GA framework.
        Default: 0.02   =>  Meaning 2% of population_size

    max_elitism_size: float
        A float value representing the fraction of maximum elitism size of the GA framework.
        Default: 0.2   =>  Meaning 20% of population_size

    convergence_limit: float
        A float value representing the convergence limit of the GA framework.
        A 0.1 value means that the algorithm will stop if there are no better solutions found
        in (0.1 * num_generations) generations.
        Default: 0.1

    use_exploration: bool
        A boolean value representing whether the GA uses exploration or not.
        Recommended value is False.
        Default: False

    save_run_info: bool
        A boolean value representing whether to save the run information.
        default: True

    """

    def __init__(self,parent_directory,warehouse_name,population_size=100,num_generations=1000,
                 hamming_attempts=1000,
                 mutation_rate_min_start=0.1,mutation_rate_min_end=0.01,
                 mutation_rate_max_start=0.3,mutation_rate_max_end=0.05,
                 crossover_rate=0.8,tournament_size_min=2,tournament_size_max=20,
                 min_elitism_size = 0.02,max_elitism_size = 0.2,
                 convergence_limit=0.1,use_exploration=False,save_run_info=True):


        self.parent_directory = parent_directory
        self.warehouse_name = warehouse_name
        self.population_size = population_size
        self.num_generations = num_generations
        self.hamming_attempts = hamming_attempts
        self.mutation_rate_min_start = mutation_rate_min_start
        self.mutation_rate_min_end = mutation_rate_min_end
        self.mutation_rate_max_start = mutation_rate_max_start
        self.mutation_rate_max_end = mutation_rate_max_end
        self.crossover_rate = crossover_rate
        self.tournament_size_min = tournament_size_min
        self.tournament_size_max = tournament_size_max
        self.min_elitism_size = int(min_elitism_size * population_size)
        self.max_elitism_size = int(max_elitism_size * population_size)
        self.use_exploration = use_exploration
        self.save_run_info = save_run_info
        self.convergence_limit = int(num_generations * convergence_limit)

        self.warehouse = None
        self.graph = None
        self.terminals = None
        self.distance_matrix = None

        self.best_solution = None
        self.best_fitness = float('inf')

        self.best_fitness_gen = [] # For monitoring, can be deleted
        self.average_fitness_gen = []   # For monitoring, can be deleted
        self.std_dev_fitness_gen = []   # For monitoring, can be deleted

        self.average_hamming_distance_gen = [] # For monitoring, can be deleted

        self.progress_gen = 0
        self.generation_counter = 0
        self.convergence_counter = 0

        self.population = []


    def import_pickup_scenario(self,pickup_scenario):

        """
        Method is used to import a specified warehouse into a format used for preprocessing steps such as:
            - Creating a graph from warehouse
            - Creating terminals
            - Creating a distance matrix

        Parameters:
        ----------

        self: FindPath

        pickup_scenario: str
            Name of the pick-up scenario with pick-up locations on which the algorithm will be applied.

        Returns:
        ---------
        None

        """
        file_path = os.path.join(self.parent_directory,self.warehouse_name,"scenarios",
                                 pickup_scenario,f"{pickup_scenario}.txt")
        #file_path = f"{self.parent_directory}/{warehouse_directory}/{warehouse_directory}.txt"
        try:
            raw_data = np.loadtxt(file_path,dtype=int)
            warehouse = np.where(raw_data == -1, float("inf"), raw_data.astype(float))
            self.warehouse = warehouse
            return None
        except Exception as e:
            print(f"Unable to load warehouse from: {file_path}   Error: {e}")
            return None

    def create_graph(self):

        """
        A helper method that is responsible for creating a graph from warehouse on which a distance matrix is created.

        Parameters:
        -----------

        self: FindPath


        Returns:
        -----------
        graph: dict[tuple[int, int], list[tuple[int, int]]]
        """

        rows, cols = self.warehouse.shape
        graph = {}
        for r in range(rows):
            for c in range(cols):
                if self.warehouse[r][c] != 1 and self.warehouse[r][c] != float("inf") and self.warehouse[r][ c] != 2:
                    neighbors = []
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), ( 0, 1)):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and self.warehouse[nr][nc] != 1 and self.warehouse[nr][nc]
                                != float("inf") and self.warehouse[nr][nc] != 2):
                            neighbors.append((nr, nc))
                    graph[(r, c)] = neighbors

        self.graph = graph

        return None

    def identify_terminals(self):
        """
        This method is used for finding coordinates of terminal nodes.
        Terminal nodes include pick-up locations and start/end point. Therefore, the length of this array is p+1

        Parameters:
        -----------

        self: FindPath

        Returns:
        -----------
        None
        """

        terminal_locations = []
        rows, cols = self.warehouse.shape
        for row_index in range(rows):
            for cols_index in range(cols):
                if self.warehouse[row_index][cols_index] == 3 or self.warehouse[row_index][cols_index] == 9:
                    terminal_locations.append((row_index, cols_index))
        self.terminals = terminal_locations

        return None

    def compute_distance_matrix(self):

        """
        This method is used to compute a distance matrix. A distance matrix is a symmetrical matrix with zero values
        along the diagonal. Each element Dij represents the shortest traversable distance between pick-up location
        i and j.
        The distance matrix allows for rapid evaluation of the total route length by candidate solutions

        Parameters:
        -----------

        self: FindPath

        Returns:
        -----------
         None
        """


        distance_matrix = np.zeros((len(self.terminals), len(self.terminals)))
        distance_matrix[~np.eye(len(self.terminals), dtype=bool)] = float("inf")
        coord_to_index = {coord: i for i, coord in enumerate(self.terminals)}

        # Naredi BFS za vsak terminal node (izdelek in start/stop)
        for i, (r0, c0) in enumerate(self.terminals):
            dist_grid = np.full_like(self.warehouse, np.inf, dtype=float)
            dist_grid[r0, c0] = 0
            queue = deque([(r0, c0)])
            remaining = len(self.terminals) - 1

            while queue and remaining > 0:
                r, c = queue.popleft()
                for nr, nc in self.graph[(r, c)]:
                    if dist_grid[nr, nc] == float("inf"):
                        dist_grid[nr, nc] = dist_grid[r, c] + 1
                        queue.append((nr, nc))
                        if (nr, nc) in coord_to_index:
                            j = coord_to_index[(nr, nc)]
                            if j != i:
                                distance_matrix[i][j] = dist_grid[nr, nc]
                                distance_matrix[j][i] = dist_grid[nr, nc]
                                remaining -= 1
        self.distance_matrix = distance_matrix

        return None

    @staticmethod
    def hamming_distance_vectorized(potential_chromosome, existing_population):
        """
        Pairwise calculation of hamming distance of potential_chromosome to every single one in existing population.

        Parameters:
        ----------

        potential_chromosome: np.ndarray Dimension: (len(self.terminals),)
            Sequence of pickup locations.

        existing_population: np.ndarray Dimension: (N,len(self.terminals))
            A set of potential solutions.


        Returns:
        --------
        np.ndarray
            An array of pairwise hamming distance with respect to potential_chromosome.
        """

        potential_chromosome = np.array(potential_chromosome)
        existing_population = np.array(existing_population)

        return np.sum(existing_population != potential_chromosome, axis=1)


    def initialize_starting_population(self):
        """
        This method is used to initialize a maximally diverse starting population with the help of Hamming distance.
        Method creates a potential solution with a random permutation.

        A potential solution is defined as a specific sequence of pick-up locations, meaning we will visit them in that
        order.
        For n pick-up locations we can visit them in n! different sequences. We want to explore that giant solution space,
        by making as different possible sequences as possible, that is why we employ a hamming distance approach for
        selecting potential solutions.

        Every randomly created chromosomes is compared to existing population and evaluated relatively to how
        different it is to other chromosomes. If it is "different enough" it is placed in the population,

        Parameters:
        -----------

        self: FindPath

        Returns:
        ----------
        None
        """

        try:
            self.population.append(list(np.random.permutation(range(1, len(self.terminals))))) # Make the first chromosome
            for _ in range(self.population_size - 1): # Make the number of chromosomes corresponding to the pop_size
                optimal_hamming_distance = len(self.terminals)

                while True: # Until we find a potential chromosome
                    is_found = False

                    for i in range(self.hamming_attempts):
                        potential_chromosome = list(np.random.permutation(range(1, len(self.terminals))))
                        hamming_distances = self.hamming_distance_vectorized(potential_chromosome, self.population)
                        min_hamming_distance = min(hamming_distances)

                        if min_hamming_distance >= optimal_hamming_distance:
                            is_found = True
                            self.population.append(potential_chromosome)
                            print(f"Number of chromosomes in initial population : {len(self.population)}")
                            break
                    if is_found:
                        break
                    else:
                        optimal_hamming_distance = optimal_hamming_distance - 1

            return None

        except Exception as e:
            print(f"There has been an error in initializing the starting population. Error: {e}")
            return None

    def compute_fitness(self, chromosome):
        """
        Computes the fitness of a single chromosome. Fitness is defined as path length of the chromosome.

        Parameters:
        -----------

        self: FindPath

        chromosome:
            np.ndarray



        Returns:
        -----------
        fitness: int
            Fitness of the chromosome.
        """

        total_distance = self.distance_matrix[0][chromosome[0]]
        for i in range(len(chromosome) - 1):
            total_distance += self.distance_matrix[chromosome[i]][chromosome[i + 1]]
        total_distance += self.distance_matrix[chromosome[-1]][0]
        return total_distance

    def get_dynamic_elitism(self,gen):
        """
        Method is used to dynamically adjust the number of chromosomes that are reproduced unchanged into the next
        generation.
        It follows a second order polynomial, ensuring lower values at first but rapidly rising when entering
         final stages of evolution.

        Parameters:
        -----------

        self: FindPath

        gen: int
            Number of current generation.

        Returns:
        ---------
        num_of_elites: int
            Number of chromosomes that are reproduced unchanged into the next generation.

        """

        progress = gen / self.num_generations
        num_of_elites = int(self.min_elitism_size + (self.max_elitism_size - self.min_elitism_size)
                            * math.pow(progress, 2))
        return num_of_elites

    def get_dynamic_mutation(self,gen):
        """
        Method is used to dynamically adjust the mutation rate. Mutation rate is sampled from a uniform
        distributing U(a,b), where a and b are dynamically adjusted.
        Favoring a larger mutation rate at early stages and lowering it in the latter stages.

        Parameters:
        -----------

        self: FindPath

        gen: int
            Number of current generation.

        Returns:
        ---------
        mutation_rate: float
            Mutation rate, a percentage of chance that a chromosome will undergo mutation

        """

        progress = gen / self.num_generations
        a = self.mutation_rate_min_start + (self.mutation_rate_min_end - self.mutation_rate_min_start) * progress
        b = self.mutation_rate_max_start + (self.mutation_rate_max_end - self.mutation_rate_max_start) * progress

        return random.uniform(a, b)

    def get_dynamic_tournament(self,gen):
        """
        Method is used to dynamically adjust the number of chromosomes that participate in the selection tournament.
        To enforce selection pressure it raises as the generations are nearing the total number of generations.

        Parameters:
        -----------

        self: FindPath

        gen: int
            Number of current generation.

        Returns:
        ---------
        tournament_size: int
            number of chromosomes that participate in the selection tournament

        """

        progress = gen / self.num_generations
        tournament_size = int(self.tournament_size_min +
                              (self.tournament_size_max - self.tournament_size_min)
                              * progress)

        return tournament_size

    def tournament_selection(self,tournament_size,fitness):
        """
        Performs tournament selection. The winner is the chromosome with the lowest fitness.

        Parameters:
        -----------

        tournament_size: int
            number of chromosomes that participate in the selection tournament

        Returns:
        -----------
        best_chromosome: np.ndarray
            Chromosome with best fitness

        """

        selected = random.sample(list(zip(self.population, fitness)), tournament_size)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    @staticmethod
    def ordered_crossover(parent_1,parent_2):
        """
        Does a standard crossover between parent_1 and parent_2.
        Selects two random positions from parent_1 array, copies everything in between those two positions
        and fills the rest with parent_2 genes, skipping duplicates

        Parameters:
        -----------

        parent_1: np.ndarray
            A chromosome

        parent_2: np.ndarray
            A chromosome

        Returns:
        -----------
        child: np.ndarray
            A child chromosome made with ordered crossover.
        """

        size = len(parent_1)
        start, end = sorted(random.sample(range(size), 2)) # Get two random numbers
        child = [None] * size # Initialize empty array with fixed size
        child[start:end + 1] = parent_1[start:end + 1] # Copy from parent_1
        available = [x for x in parent_2 if x not in child[start:end + 1]] # find missing values in parent_1
        avail_index = 0
        for i in range(size):
            if child[i] is None:
                child[i] = available[avail_index]
                avail_index += 1
        return child

    @staticmethod
    def single_gene_mutation(chromosome):

        """
        Does a single gene swap.

        Parameters:
        -----------

        chromosome: np.ndarray
            A chromosome


        Returns:
        -----------
        mutated: np.ndarray
            A mutated chromosome.
        """

        mutated = chromosome.copy()
        i, j = sorted(random.sample(range(len(chromosome)), 2))
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    @staticmethod
    def two_opt_swap_mutation(chromosome):
        """
        Does a two-opt swap, a frequently used technique for TSP like problems.

        Parameters:
        -----------

        chromosome: np.ndarray
            A chromosome

        Returns:
        -----------
        mutated: np.ndarray
            A mutated chromosome.
        """

        i, j = sorted(random.sample(range(len(chromosome)), 2))
        mutated = chromosome[:i] + list(reversed(chromosome[i:j + 1])) + chromosome[j + 1:]
        return mutated

    @staticmethod
    def calc_hamming_distance(chromosome_1, chromosome_2):
        """
        Calculates hamming distance of two chromosomes.
        It is a measure of dissimilarity between two potential solutions.

        Parameters:
        -----------

        chromosome_1: np.ndarray
            A chromosome

        chromosome_2: np.ndarray
            A chromosome

        Returns:
        --------

        hamming_distance: int
            A measure of dissimilarity between two potential solutions.
        """

        hamming_distance = 0
        for i in range(len(chromosome_1)):
            if chromosome_1[i] != chromosome_2[i]:
                hamming_distance += 1
        return hamming_distance


    def bfs_segment(self,start_node,end_node):
        """
        Method is used to find the shortest path between two pick-up locations.

        Parameters:
        -----------
        self: FindPath

        start_node: int
            Starting pick-up location
        end_node: int
            Ending pick-up location

        Returns:
        ----------
        path: np.ndarray

        """

        queue = deque([[start_node]])
        visited = set()
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == end_node:
                return path
            for neighbor in self.graph.get(current,[]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []


    def draw_path(self, path):
        warehouse = self.warehouse.tolist()
        num_step = -1
        for step in path:
            if not (warehouse[step[0]][step[1]] == 3 or warehouse[step[0]][step[1]] == 9 or warehouse[step[0]][
                step[1]] == 0):  # Smo ga ze visital
                warehouse[step[0]][step[1]] = warehouse[step[0]][step[1]] + "," + str(num_step)
                num_step -= 1
            else:  # Prvic visital
                warehouse[step[0]][step[1]] = str(warehouse[step[0]][step[1]]) + "->" + str(num_step)
                num_step -= 1

        str_warehouse = [[str(col) for col in row] for row in warehouse]
        max_string_width = max(len(col) for row in str_warehouse for col in row)

        for row in warehouse:
            print(" | ".join(f"{str(col):>{max_string_width}}" for col in row))
        return None


    def find_path(self,pickup_scenario):
        """
        Method is used to find an optimal path for a specific pick-up scenario. It will provide a json file,
        if the save_run_info is set to True.
        With given results:
            - path_length: of the shortest path found
            - full path: a list of "steps" that the AGV should take to achieve this path.
            - best_solution: Sequence of pick-up locations to visit for the shortest path.

        Parameters:
        -----------
        self: FindPath

        pickup_scenario: str
            Name of the pick-up scenario with pick-up locations on which the algorithm will be applied.

        Returns:
        --------
        None
        """

        self.import_pickup_scenario(pickup_scenario)
        self.create_graph()
        self.identify_terminals()
        self.compute_distance_matrix()


        self.initialize_starting_population() # initializes diverse starting population with HD

        for gen in range(self.num_generations): # Evolutionary loop

            # The following blocks correspond to the evaluation of population and saving better solutions

            # Compute fitness for the population
            fitness = [self.compute_fitness(chromosome) for chromosome in self.population]
            # Save to a tuple chromosome and its corresponding fitness
            combined = list(zip(self.population, map(int, fitness)))
            # Sort into ascending order, lowest fitness is the first element
            combined.sort(key=lambda x: x[1])
            # Save the best fitness of this generation MONITORING
            self.best_fitness_gen.append(combined[0][1])
            # Save average fitness of this generation MONITORING
            self.average_fitness_gen.append(np.mean(fitness))
            # Save standard deviation fitness of this generation MONITORING
            self.std_dev_fitness_gen.append(np.std(fitness))

            # If we didn't find a better solution
            if combined[0][1] >= self.best_fitness:
                self.convergence_counter += 1
                if self.convergence_counter >= self.convergence_limit:
                    break

            # We found a better solution
            else:
                self.best_fitness = combined[0][1]
                self.best_solution = combined[0][0]
                self.convergence_counter = 0

            # The following blocks correspond the genetic operations that create the next generation
            new_population = [] # Initialize empty array for the next generation

            # Elitism
            elite = combined[:self.get_dynamic_elitism(gen)] # Get the top individuals and put them in elite
            new_population.extend([ind for ind, fit in elite]) # Just copy the chromosomes, no need for fitness

            # Next blocks are used for filling the creating and filling the population with standard genetic operations

            # Mutation parameters
            mutation_rate = self.get_dynamic_mutation(gen)

            # Until we fill the population
            while len(new_population) < self.population_size:

                # Procreation
                parent_1 = self.tournament_selection(self.get_dynamic_tournament(gen),fitness) # Get the first parent

                # Crossover
                if random.random() < self.crossover_rate:
                    # Get second parent, ensure it is different
                    while True:
                        parent_2 = self.tournament_selection(self.get_dynamic_tournament(gen),fitness)
                        if not np.array_equal(parent_1, parent_2):
                            break

                    child_1 = self.ordered_crossover(parent_1,parent_2)
                    child_2 = self.ordered_crossover(parent_2,parent_1)
                    new_population.append(child_1)
                    new_population.append(child_2)

                # Reproduction
                else:
                    child = parent_1.copy()
                    new_population.append(child)

                # Mutation
                if random.random() < mutation_rate:
                    chromosome_index = random.randint(0, len(new_population) - 1)
                    try:
                        # Single gene mutation
                        if random.random() <= 0.4:
                            new_population[chromosome_index] = self.single_gene_mutation(new_population[chromosome_index])
                        # Two-opt swap
                        else:
                            new_population[chromosome_index] = self.two_opt_swap_mutation(new_population[chromosome_index])
                    except Exception as e:
                        print(f"Error when mutating chromosome {chromosome_index}")

            self.population = new_population # Store the new population

            # Calculate and store average hamming distance of population

            num_of_pairwise_hamm_dist = (self.population_size * (self.population_size - 1)) / 2 # Gaussian sum, always int
            average_hamm_dist  = 0
            for i in range(self.population_size):
                for j in range(i+1,self.population_size):
                    average_hamm_dist += self.calc_hamming_distance(self.population[i],self.population[j])

            average_hamm_dist = average_hamm_dist / num_of_pairwise_hamm_dist
            self.average_hamming_distance_gen.append(average_hamm_dist)


            print(f"#------------------------------Generation : {gen}---------------------------------------------#")
            print(f"Best Fitness: {self.best_fitness}")
            print(f"Average Hamming Distance: {average_hamm_dist:.2f}")
            print("#----------------------------------------------------------------------------------------------#")

        print(f"#---------------------------------Found optimal path----------------------------------------------#")
        print(f"Shortest tour length: {self.best_fitness}")
        print(f"#-------------------------------------------------------------------------------------------------#")




        #Creates a list of steps (x,y) where every (x,y) is the position of the empty block
        self.best_solution = [0] + list(self.best_solution) + [0] # It begins and ends at the start node
        full_path = []
        for i in range(len(self.best_solution) - 1):
            start_node = self.terminals[self.best_solution[i]]
            end_node = self.terminals[self.best_solution[i+1]]
            segment = self.bfs_segment(start_node, end_node)

            if i != 0:
                segment = segment[1:]
            full_path.extend(segment)

        # Convert to int
        self.best_solution = [int(element) for element in self.best_solution]

        # Following block is responsibly for saving the run info in a json format.
        if self.save_run_info:

            run_data = {
                "path_length": int(self.best_fitness),
                "full_path": full_path,
                "best_solution": self.best_solution,
            }

            #print(self.draw_path(full_path))
            json_file = os.path.join(self.parent_directory,self.warehouse_name,"scenarios",
                                 pickup_scenario,"results")
            os.makedirs(json_file, exist_ok=False)
            json_file = os.path.join(json_file,"results.json")
            try:
                with open(json_file, "w") as json_file:
                    json.dump(run_data, json_file, sort_keys=False)
            except Exception as e:
                print(f"Error when saving results to {json_file} Error: {e}")

        return None







