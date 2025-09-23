import numpy as np
import random
from collections import deque
import random
import math


"""
This is a Genetic Algorithm based framework for warehouse navigation,
modelled as a Travelling Salesman Problem (TSP) variant for Automated Guided Vehicles (AGVs).
The warehouse layout is represented as a graph, where pick-up locations serve as terminal nodes. 
A distance matrix, compute-ed via Breadth-First Search (BFS) enables efficient route evaluation. 
To promote diversity in the initial population, a Hamming distance-based vectorized initialization strategy is employed,
ensuring that the chromosomes are maximally distinct. 
The GA balances exploration and exploitation by dynamically adjusting the fitness function. 
Early generations emphasize diversity, while later ones focus on solution refinement, 
improving convergence and avoiding premature stagnation.

GA based solutions perform significantly better when potential solutions (chromosomes) are maximally distinct
and perform poorly when they are similar or equal. Therefore a Hamming distance is used to ensure diversity in the
initial population. While there is an option to segment the algorithm into exploration/exploitation it is usually not
necessary, unless you are willing to sacrifice extra computational time. I would advise just using exploitation.

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

    ----------
    Parameters:
    ----------
        self: FindPath
            A FindPath instance that contains the GA framework used to compute optimal path.

        parent_directory: str
            Name of the parent directory in which the warehouses are stored.

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
            Default: 0.1.

        use_exploration: bool
            A boolean value representing whether the GA uses exploration or not.
            Recommended value is False.
            Default: False

        save_run_info: bool
            A boolean value representing whether to save the run information.
            default: True

    """

    def __init__(self,parent_directory,population_size=100,num_generations=1000,
                 mutation_rate_min_start=0.1,mutation_rate_min_end=0.01,
                 mutation_rate_max_start=0.3,mutation_rate_max_end=0.05,
                 crossover_rate=0.8,tournament_size_min=2,tournament_size_max=20,
                 min_elitism_size = 0.02,max_elitism_size = 0.2,
                 convergence_limit=0.1,use_exploration=False,save_run_info=True):


        self.parent_directory = parent_directory
        self.population_size = population_size
        self.num_generations = num_generations
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
        self.terminals = None
        self.distance_matrix = None

    def import_warehouse(self,warehouse_directory):

        """
        Method is used to import a specified warehouse into a format used for preprocessing steps such as:
            - Creating a graph from warehouse
            - Creating terminals
            - Creating a distance matrix

        ----------
        Parameters:
        ----------

        self: FindPath

        warehouse_directory: str
            Name of the warehouse with pick-up locations on which the algorithm will be applied.

        ---------
        Returns: None
        ---------

        """

        file_path = f"{self.parent_directory}/{warehouse_directory}/{warehouse_directory}.txt"
        try:
            raw_data = np.loadtxt(file_path,dtype=int)
            warehouse = np.where(raw_data == -1, float("inf"), raw_data.astype(float))
            self.warehouse = warehouse
        except Exception as e:
            print(f"Unable to load warehouse from: {file_path}   Error: {e}")
            return None

    def create_graph(self):

        """
        A helper method that is responsible for creating a graph from warehouse on which a distance matrix is created.

        -----------
        Parameters:
        -----------

        self: FindPath

        -----------
        Returns:
        graph: dict[tuple[int, int], list[tuple[int, int]]]
        -----------
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
        return graph

    def identify_terminals(self):
        """
        This method is used for finding coordinates of terminal nodes.
        Terminal nodes include pick-up locations and start/end point. Therefore, the length of this array is p+1

        -----------
        Parameters:
        -----------

        self: FindPath

        -----------
        Returns: None
        -----------
        """

        terminal_locations = []
        rows, cols = self.warehouse.shape
        for row_index in range(rows):
            for cols_index in range(cols):
                if self.warehouse[row_index][cols_index] == 3 or self.warehouse[row_index][cols_index] == 9:
                    terminal_locations.append((row_index, cols_index))
        self.terminals = terminal_locations

    def compute_distance_matrix(self,graph):
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
                for nr, nc in graph[(r, c)]:
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




