import itertools

import numpy as np
import copy
import time
import itertools
import random


class Sudoku_grid:

    def __init__(self, grid=None):
        if grid is None:
            raise ValueError("Grid Initial state not provided")
        if not isinstance(grid, np.ndarray):
            raise ValueError("Grid initial state must be provided as a 9x9 numpy array")
        for rows in grid:
            for col in rows:
                if not (isinstance(col, np.int8) or isinstance(col, np.int16) or isinstance(col, np.int32) or isinstance(col, np.int64)):
                    raise ValueError("Cell value must be integers")
                if col > 9 or col < 0:
                    raise ValueError("Value out ouf range: Each cell must have a value between 0 and 9 included")
        A = [set([j for j in range(1, 10)]) for i in range(1, 10)]
        B = [set([j for j in range(1, 10)]) for i in range(1, 10)]
        C = [set([j for j in range(1, 10)]) for i in range(1, 10)]
        D = [set([j for j in range(1, 10)]) for i in range(1, 10)]
        E = [set([j for j in range(1, 10)]) for i in range(1, 10)]
        F = [set([j for j in range(1, 10)]) for i in range(1, 10)]
        G = [set([j for j in range(1, 10)]) for i in range(1, 10)]
        H = [set([j for j in range(1, 10)]) for i in range(1, 10)]
        I = [set([j for j in range(1, 10)]) for i in range(1, 10)]

        for idx_row, row in enumerate([A, B, C, D, E, F, G, H, I]):
            for idx_col, col in enumerate(row):
                val = int(grid[idx_row][idx_col])
                if val != 0:
                    row[idx_col] = {val}

        A1, A2, A3, A4, A5, A6, A7, A8, A9 = tuple(A)
        B1, B2, B3, B4, B5, B6, B7, B8, B9 = tuple(B)
        C1, C2, C3, C4, C5, C6, C7, C8, C9 = tuple(C)
        D1, D2, D3, D4, D5, D6, D7, D8, D9 = tuple(D)
        E1, E2, E3, E4, E5, E6, E7, E8, E9 = tuple(E)
        F1, F2, F3, F4, F5, F6, F7, F8, F9 = tuple(F)
        G1, G2, G3, G4, G5, G6, G7, G8, G9 = tuple(G)
        H1, H2, H3, H4, H5, H6, H7, H8, H9 = tuple(H)
        I1, I2, I3, I4, I5, I6, I7, I8, I9 = tuple(I)

        #Rows (top to bottom)
        Row1 = [A1, A2, A3, A4, A5, A6, A7, A8, A9]
        Row2 = [B1, B2, B3, B4, B5, B6, B7, B8, B9]
        Row3 = [C1, C2, C3, C4, C5, C6, C7, C8, C9]
        Row4 = [D1, D2, D3, D4, D5, D6, D7, D8, D9]
        Row5 = [E1, E2, E3, E4, E5, E6, E7, E8, E9]
        Row6 = [F1, F2, F3, F4, F5, F6, F7, F8, F9]
        Row7 = [G1, G2, G3, G4, G5, G6, G7, G8, G9]
        Row8 = [H1, H2, H3, H4, H5, H6, H7, H8, H9]
        Row9 = [I1, I2, I3, I4, I5, I6, I7, I8, I9]

        #Columns (left to rigth)
        Col1 = [A1, B1, C1, D1, E1, F1, G1, H1, I1]
        Col2 = [A2, B2, C2, D2, E2, F2, G2, H2, I2]
        Col3 = [A3, B3, C3, D3, E3, F3, G3, H3, I3]
        Col4 = [A4, B4, C4, D4, E4, F4, G4, H4, I4]
        Col5 = [A5, B5, C5, D5, E5, F5, G5, H5, I5]
        Col6 = [A6, B6, C6, D6, E6, F6, G6, H6, I6]
        Col7 = [A7, B7, C7, D7, E7, F7, G7, H7, I7]
        Col8 = [A8, B8, C8, D8, E8, F8, G8, H8, I8]
        Col9 = [A9, B9, C9, D9, E9, F9, G9, H9, I9]

        #Squares (left to right, top to bottom)
        Sqr1 = [A1, A2, A3, B1, B2, B3, C1, C2, C3]
        Sqr2 = [A4, A5, A6, B4, B5, B6, C4, C5, C6]
        Sqr3 = [A7, A8, A9, B7, B8, B9, C7, C8, C9]
        Sqr4 = [D1, D2, D3, E1, E2, E3, F1, F2, F3]
        Sqr5 = [D4, D5, D6, E4, E5, E6, F4, F5, F6]
        Sqr6 = [D7, D8, D9, E7, E8, E9, F7, F8, F9]
        Sqr7 = [G1, G2, G3, H1, H2, H3, I1, I2, I3]
        Sqr8 = [G4, G5, G6, H4, H5, H6, I4, I5, I6]
        Sqr9 = [G7, G8, G9, H7, H8, H9, I7, I8, I9]

        self.units = [Row1, Row2, Row3, Row4, Row5, Row6, Row7, Row8, Row9,
                      Col1, Col2, Col3, Col4, Col5, Col6, Col7, Col8, Col9,
                      Sqr1, Sqr2, Sqr3, Sqr4, Sqr5, Sqr6, Sqr7, Sqr8, Sqr9]

    def is_valid(self):
        for unit in self.units:
            if set() in unit:
                return False
        return True

    def is_invalid(self):
        for unit in self.units:
            if set() in unit:
                return True
        return False

    def is_gloal(self):
        for unit in self.units:
            if sum(Sudoku_grid.get_len_domain(unit)) != 9:
                return False
        return True

    def get_grid(self):
        grid = []
        rows = copy.deepcopy(self.units[0:9])
        for row in rows:
            len_domain = Sudoku_grid.get_len_domain(row)
            for idx, n in enumerate(len_domain):
                if n > 1:
                    row[idx] = {0}
            grid.append([item for t in row for item in t])
        return np.array(grid)

    def set_failed_grid(self):
        for unit in self.units[0:9]:
            for domain in unit:
                domain.difference_update(domain)
                domain.add(-1)

    @staticmethod
    def get_len_domain(unit):
        return [len(s) for s in unit]


class Sudoku_game():
    def __init__(self):
        self.explored_nodes = 0;
        self.tree_depth = 0
        self.constrain_eval = 0
        self.unit_map = {0: [[9, 18], [10, 18], [11, 18], [12, 19], [13, 19], [14, 19], [15, 20], [16, 20], [17, 20]],
                         1: [[9, 18], [10, 18], [11, 18], [12, 19], [13, 19], [14, 19], [15, 20], [16, 20], [17, 20]],
                         2: [[9, 18], [10, 18], [11, 18], [12, 19], [13, 19], [14, 19], [15, 20], [16, 20], [17, 20]],
                         3: [[9, 21], [10, 21], [11, 21], [12, 22], [13, 22], [14, 22], [15, 23], [16, 23], [17, 23]],
                         4: [[9, 21], [10, 21], [11, 21], [12, 22], [13, 22], [14, 22], [15, 23], [16, 23], [17, 23]],
                         5: [[9, 21], [10, 21], [11, 21], [12, 22], [13, 22], [14, 22], [15, 23], [16, 23], [17, 23]],
                         6: [[9, 24], [10, 24], [11, 24], [12, 25], [13, 25], [14, 25], [15, 26], [16, 26], [17, 26]],
                         7: [[9, 24], [10, 24], [11, 24], [12, 25], [13, 25], [14, 25], [15, 26], [16, 26], [17, 26]],
                         8: [[9, 24], [10, 24], [11, 24], [12, 25], [13, 25], [14, 25], [15, 26], [16, 26], [17, 26]],
                         9: [[0, 18], [1, 18], [2, 18], [3, 21], [4, 21], [5, 21], [6, 24], [7, 24], [8, 24]],
                         10: [[0, 18], [1, 18], [2, 18], [3, 21], [4, 21], [5, 21], [6, 24], [7, 24], [8, 24]],
                         11: [[0, 18], [1, 18], [2, 18], [3, 21], [4, 21], [5, 21], [6, 24], [7, 24], [8, 24]],
                         12: [[0, 19], [1, 19], [2, 19], [3, 22], [4, 22], [5, 22], [6, 25], [7, 25], [8, 25]],
                         13: [[0, 19], [1, 19], [2, 19], [3, 22], [4, 22], [5, 22], [6, 25], [7, 25], [8, 25]],
                         14: [[0, 19], [1, 19], [2, 19], [3, 22], [4, 22], [5, 22], [6, 25], [7, 25], [8, 25]],
                         15: [[0, 20], [1, 20], [2, 20], [3, 23], [4, 23], [5, 23], [6, 26], [7, 26], [8, 26]],
                         16: [[0, 20], [1, 20], [2, 20], [3, 23], [4, 23], [5, 23], [6, 26], [7, 26], [8, 26]],
                         17: [[0, 20], [1, 20], [2, 20], [3, 23], [4, 23], [5, 23], [6, 26], [7, 26], [8, 26]],
                         18: [[0, 9], [1, 9], [2, 9], [0, 10], [1, 10], [2, 10], [0, 11], [1, 11], [2, 11]],
                         19: [[0, 12], [1, 12], [2, 12], [0, 13], [1, 13], [2, 13], [0, 14], [1, 14], [2, 14]],
                         20: [[0, 15], [1, 16], [2, 17], [0, 15], [1, 16], [2, 17], [0, 15], [1, 16], [2, 17]],
                         21: [[3, 9], [3, 10], [3, 11], [4, 9], [4, 10], [4, 11], [5, 9], [5, 10], [5, 11]],
                         22: [[3, 12], [3, 13], [3, 14], [4, 12], [4, 13], [4, 14], [5, 12], [5, 13], [5, 14]],
                         23: [[3, 15], [3, 16], [3, 17], [4, 15], [4, 16], [4, 17], [5, 15], [5, 16], [5, 17]],
                         24: [[6, 9], [6, 10], [6, 11], [7, 9], [7, 10], [7, 11], [8, 9], [8, 10], [8, 11]],
                         25: [[6, 12], [6, 13], [6, 14], [7, 12], [7, 13], [7, 14], [8, 12], [8, 13], [8, 14]],
                         26: [[6, 15], [6, 16], [6, 17], [7, 15], [7, 16], [7, 17], [8, 15], [8, 16], [8, 17]]}

    def ac_3(self, state, first_unit_idx=None, var_idx=None):
        # idx = self.get_best_unit_priority(state)
        if first_unit_idx is None:
            unit_queue = [i for i in range(27)]  # put all unit in the queue
        else:
            unit_queue = self.unit_map[first_unit_idx][var_idx] + [first_unit_idx]
        eval = 0
        print(f"Initial grid State")
        print(state.get_grid())
        while len(unit_queue) > 0:
            eval += 1
            unit_idx = unit_queue.pop(0)
            unit = state.units[unit_idx]
            start_unit = copy.deepcopy(unit)
            print(f"Initial state unit {unit_idx} ")
            print(unit)
            self.naked_single(unit)
            if unit != start_unit:
                print(f"Unit state after running naked_single on unit number {unit_idx}")
                print(unit)
                print()
            unit_temp = copy.deepcopy(unit)
            self.hidden_single(unit)
            if unit != unit_temp:
                print(f"Unit state after running hidden_single on unit number {unit_idx}")
                print(unit)
                print()
            unit_temp = copy.deepcopy(unit)
            self.naked_double(unit)
            if unit != unit_temp:
                print(f"Unit state after running naked_double on unit number {unit_idx}")
                print(unit)
                print()
            # Run "hidden double" only if the other constraints did not shrink the domain as it is computationally expensive
            if unit == unit_temp:
                self.hidden_double(unit)
                unit_temp = copy.deepcopy(unit)
                if unit != unit_temp:
                    print(f"Unit state after running hidden double on unit number {unit_idx}")
                    print(unit)
                    print()
            if not state.is_valid():
                return eval
            if unit != start_unit:
                this_unit_map = self.unit_map[unit_idx]
                for idx, dom in enumerate(unit):
                    if dom != start_unit[idx]:
                        units_2_add = this_unit_map[idx]
                        for new_unit in units_2_add:
                            if new_unit not in unit_queue:
                                unit_queue.append(new_unit)  # adding all units linked to this unit  but in theory i could  look what domains have changed and only get the right indexes
        return eval

    def naked_single(self, unit):
        fun = lambda u, val, i: [s.difference_update(val) for idx, s in enumerate(u) if idx != i]
        for idx, domain in enumerate(unit):
            if len(domain) == 1:
                fun(unit, domain, idx)

    def hidden_single(self, unit):
        # fun = lambda set_test, u, i: [set_test.union(s) for idx, s in enumerate(u) if idx != i]
        for idx, domain in enumerate(unit):
            if len(domain) > 1:
                s_test_1 = Sudoku_game.get_test_set_hidden_single(unit, idx)
                if len(domain.difference(s_test_1)) == 1:
                    domain.difference_update(s_test_1)

    def naked_double(self, unit):
        fun = lambda u, val: [s.difference_update(val) for s in u if s != val]
        for idx, domain in enumerate(unit):
            if len(domain) == 2:
                if domain in unit[idx + 1:-1]:
                    fun(unit, domain)

    def hidden_double(self, unit):
        for idx, domain in enumerate(unit):
            if 2 < len(domain) <= 5:
                hidden_pairs = itertools.combinations(list(domain),2)
                for a,b in hidden_pairs:
                    a_set = {a,}
                    b_set = {b,}
                    a_set_location = [1 if a_set.issubset(s) and len(s) >= 2 else 0 for s in unit]
                    b_set_location = [1 if b_set.issubset(s) and len(s) >= 2 else 0 for s in unit]
                    if a_set_location == b_set_location and sum(a_set_location) == 2:
                        pair_set = a_set.union(b_set)
                        domain.difference_update(domain-pair_set)
                        a_set_location[idx] = 0
                        idx_next_domain = a_set_location.index(1)
                        next_domain = unit[idx_next_domain]
                        next_domain.difference_update(next_domain-pair_set)
                        return

    def backtracking(self, state):
        self.explored_nodes = 0
        result, depth = self.backtracking_search(state)
        if depth < 0:
            depth = 0
        else:
            depth -= 1
        self.tree_depth = depth
        return result

    def backtracking_ver1(self, state):
        self.explored_nodes = 0
        self.ac_3(state)
        result, depth = self.backtracking_search_ver1(state)
        if depth < 0:
            depth = 0
        else:
            depth -= 1
        self.tree_depth = depth
        return result

        # Backtracking depth-first search

    def backtracking_search_ver1(self, starting_state):
        self.explored_nodes += 1
        if starting_state.is_invalid():  # Check in consistency is maintained
            return None, -1
        if starting_state.is_gloal():  # Check if this is the goal
            return starting_state, 1
        actions = self.get_actions(starting_state)  # The next best actions inference
        for action in actions:
            tree_depth = 1
            next_state = self.apply_assignment(starting_state, action)  # make assignment and get next state space
            self.ac_3(next_state, action[0], action[1])  # Enforce the constraints

            next_state, incremental_depth = self.backtracking_search_ver1(next_state)  # explore the next state space
            tree_depth += incremental_depth
            if next_state is not None and next_state.is_gloal():  # return if the goal has been found.
                return next_state, tree_depth
        return None, incremental_depth

    # Backtracking depth-first search
    def backtracking_search(self, starting_state):
        self.ac_3(starting_state)  # Enforce the constraints
        self.explored_nodes += 1
        if starting_state.is_invalid():  # Check in consistency is maintained
            return None, -1
        if starting_state.is_gloal():  # Check if this is the goal
            return starting_state, 1
        actions = self.get_actions(starting_state)  # The next best actions inference
        for action in actions:
            tree_depth = 1
            next_state = self.apply_assignment(starting_state, action)  # make assignment and get next state space
            next_state, incremental_depth = self.backtracking_search(next_state)  # explore the next state space
            tree_depth += incremental_depth
            if next_state is not None and next_state.is_gloal():  # return if the goal has been found.
                return next_state, tree_depth
        return None, incremental_depth


    def get_actions(self, state):

        actions = []
        units_priority = [sum(Sudoku_grid.get_len_domain(unit)) if sum(Sudoku_grid.get_len_domain(unit)) > 9 else 1000
                          for unit in state.units]
        best_unit_idx = min(range(len(units_priority)),
                            key=units_priority.__getitem__)  # unit that has the smallest number of free variable in the domain
        best_unit = state.units[best_unit_idx]
        var_priority = [len(var) if len(var) > 1 else 1000 for var in best_unit]
        best_var_idx = min(range(len(var_priority)),
                           key=var_priority.__getitem__)  # varialbe that has the smallest number of free values in the domain
        values = list(best_unit[best_var_idx])
        value_scores = []
        sorted_values_by_score = []
        for val in values:
            val_num = 0
            for unit in state.units[0:9]:
                val_num += sum([1 if val in domain else 0 for domain in unit])
            value_scores.append(val_num)
        for i in range(len(value_scores)):
            idx_max_val = min(range(len(value_scores)), key=value_scores.__getitem__)
            sorted_values_by_score.append(values[idx_max_val])
            value_scores[idx_max_val] = 1000

        for val in sorted_values_by_score:
            actions.append((best_unit_idx, best_var_idx, val))
        return actions

    def apply_assignment(self, state, action):
        unit_idx, var_idx, value = action
        next_state = copy.deepcopy(state)
        unit = next_state.units[unit_idx]
        unit[var_idx].intersection_update([value])
        return next_state

    def solve(self, state):
        result = self.backtracking_ver1(state)
        solFound = True
        if result is None:
            result = Sudoku_grid(np.array([[0 for _ in range(9)] for _ in range(9)]))
            result.set_failed_grid()
            solFound = False
        print(f"Tree depth = {self.tree_depth}")
        print(f"Explored Node = {self.explored_nodes}")
        return solFound, result


    @staticmethod
    def get_test_set_hidden_single(domain, idx_avoid):
        s_test = set()
        for idx, s in enumerate(domain):
            if idx != idx_avoid:
                s_test = s_test.union(s)
        return s_test


def sudoku_solver(sudoku):
    state = Sudoku_grid(sdk)
    game = Sudoku_game()
    solFound, result = game.solve(state)
    calculated_solution = result.get_grid()
    return calculated_solution


if __name__ == "__main__":
    # Load sudokus
    nodes = 0
    t = 0

    for sdk_num in range(1):
        t_start = time.process_time()
        sudoku = np.load("../data/hard_puzzle.npy")
        sudoku_solutions = np.load("../data/hard_solution.npy")
        sdk = sudoku[sdk_num]
    # sdk = np.array([[7, 0, 0, 0, 0, 0, 6, 8, 1],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 9],
    #                 [0, 5, 0, 3, 0, 0, 0, 0, 0],
    #                 [8, 0, 0, 1, 0, 0, 0, 0, 0],
    #                 [0, 3, 0, 9, 0, 0, 0, 0, 4],
    #                 [1, 0, 0, 0, 0, 3, 0, 2, 5],
    #                 [0, 0, 0, 0, 2, 0, 9, 0, 0],
    #                 [0, 4, 8, 0, 0, 0, 1, 0, 0],
    #                 [0, 0, 7, 6, 9, 0, 0, 0, 0]])
    #AI escargot Sudoku
        # sdk = np.array([[1, 0, 0, 0, 0, 7, 0, 9, 0],
        #             [0, 3, 0, 0, 2, 0, 0, 0, 8],
        #             [0, 0, 9, 6, 0, 0, 5, 0, 0],
        #             [0, 0, 5, 3, 0, 0, 9, 0, 0],
        #             [0, 1, 0, 0, 8, 0, 0, 0, 2],
        #             [6, 0, 0, 0, 0, 4, 0, 0, 0],
        #             [3, 0, 0, 0, 0, 0, 0, 1, 0],
        #             [0, 4, 0, 0, 0, 0, 0, 0, 7],
        #             [0, 0, 7, 0, 0, 0, 3, 0, 0]])

    # sdk = np.array([[0, 0, 0, 4, 0, 0, 0, 8, 6],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [3, 1, 0, 0, 0, 0, 0, 2, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 1, 0, 7, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 9, 0, 0, 0, 4, 0],
    #                 [0, 0, 0, 0, 0, 0, 5, 0, 0],
    #                 [0, 3, 0, 7, 0, 0, 0, 0, 0]])

        #sdk_sol = sudoku_solutions[sdk_num]
    # print("Starting grid")
    #print(sdk)
    #print("Given Solution")
    #print(sdk_sol)
        state = Sudoku_grid(sdk)
        game = Sudoku_game()
        solFound, result = game.solve(state)

    # print(f"number of constraints evaluations {n_eval}")
        if solFound:
            print(f"Solution Found for  sudoku number {sdk_num}")
        else:
            print(f"Solution not found for  sudoku number {sdk_num}")
        calculated_solution = result.get_grid()
        #print(sdk_sol == calculated_solution)
    #print(calculated_solution)
        t_elapsed = time.process_time() - t_start
        print(f"Execution time = {t_elapsed}")
        print()
        print()
        nodes += game.explored_nodes
        t += t_elapsed
    print(f"Total number of explored nodes: {nodes}")
    print(f"Total time to solve {sdk_num+1} Sudokus is {t}")


