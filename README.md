## Introduction and Terminology ##

Before diving into the detail explanations regarding the choice of algorithm and data structure used in the assignment it is worth clarifying some of terminology used both in the code and throughout this text:
- Cell: Each single cell in the Sudoku grid which can contain one single value from 1 to 9 in the Sudoku grid.  
- Unit:  row, column or square forming the Sudoku grid.
- Cell Domain:  also referred as just domain, it identifies all possible values a single cell can take while maintaining consistency.


## Search algorithm ##
To solve the Sudoku I have implemented a Backtracking depth-first search algorithm for CSP (constrain satisfaction problem) whose sudo-code is available in [1] chapter 6.3  pp 214.
Since a valid Sudoku grid (with at least 17 initial clues) admits only one valid solution and since it presents a finite search space, this implementation of the algorithm is complete and memory, efficient presenting a linear worst-case space complexity of $O(m)$ where m is the maximum depth of the tree.

To maintain **node-consistency** on each unit and **arc-consistency** amongst all units, an AC-3 algorithm is also implemented. Its implementation is explained in great details in [1] chapter 6.2 pp 209.
The AC-3 algorithm makes use of a map which contain information about the dependency between each unit needed to schedule the appropriate units to run the constraints on. For example, if one constrain shrinks the domain of one cell, the AC-3 algorithm retrieve the information about what other units shares the same cell and add them to the queue of units to run constraints on.
This approach optimizes the total number of constraints being run while maximizing the reduction of the domain of each cell. This helps to increase the execution speed of the search algorithm.

Four (4) constraints have been implemented, more specifically:
1. Naked single: For a given unit and cell, if the cell has only one value possible in its domain then the cell must have that value.

2. Hidden single: For a given unit and cell, if the cell has multiple values in its domain but a value x is not present in any other cell domain for the same unit, then the cell must take that value x and the other values can be discarded.

3. Naked double: For any two cells c1 and c2 in the same unit, if those two cells have only the same two values x and y in their domains, then c1 and c2 must take x or y or vice versa. Therefore, x and y can be discarded from any other cell domain for the same unit.

4. Hidden double: For any two cells c1 and c2 in the same unit, if those two cells have the same two values x and y in their domains, amongst possibly other values, and those two values cannot be found in any other domain for the same units, then the x and y must belong to the cell c1 and c2, therefore all the other values in the domain of c1 and c2 can be discarded.

The first three constraints run indiscriminately on every unit being added on the queue,  the 4th constraint, being computationally expensive to enforce, is only run if the previous three did not have any effect of the domain of any variable in the unit.
This conditional solution has been adopted after seeing a noticeable reduction of the algorithm execution speed when the 4th constrain was let to run with the same frequency of the other three constraints.

When AC-3 is run on the starting state, the queue of units is initialized by adding all units to it, as no information about the latest assignment is given. During the search process and after a new state is created upon a new value assignment on a specific unit and cell, when the AC-3 algorithm is called to shrink the state space, the queue of units is initialized to only those ones that are affected by the assignment. This helps to avoid unnecessary constraint runs on units which would result in an unchanged state.

During the search, once the state space has been shrunk by the constraints, rather than selecting the next assignment at random, a heuristic function is used to estimate the next best move, this strategy is explained in chapter 6.3.1 of [1]

The used heuristic function estimates what unit, cell and value should be next assigned, effectively estimating what next state is best to explore.
More specifically, the unit and the cell that present the least number of free variables in their domain are chosen. This strategy proves to be effective as during the test performed on the hard Sudokus it minimized the number of nodes being explored during the search, See Result comparison section for more details.
Following the same principle, once the next best unit and cell have been identified, the order of the possible values is decided in such a way to prioritize those ones that are most assigned on whole grid.
Another approach which I have tested was to prioritize values which are the least assigned in whole grid, but after a comparison between the two approaches on the 15 hard Sudokus I have found that the first approach resulted in fewer nodes being explored and therefore a more efficient algorithm. Such results are shown below in the results comparison section (second table)




## Data structure ##
This section describes the data structure used to model the Sudoku grid and the motivation which led to such chooses.
Each one of the 81 cells in the Sudoku grid is modeled with a set containing all possible values of its domain e.g. {1,2,3,4,6,7,8,9}
The domain of each cell (set) is initialized at the very beginning when the Sudoku grid is instantiated. The cell domains are then referenced by the appropriate units which are kept in memory in a list by the property "units" (Rows, column and squares). Different units which contain the same cell point to the the same set in memory, therefore if one set is updated in one unit, all units that share that cell see the updated cell domain. 
The "units" property of Sudoku grid class is a list containing all 27 possible units, each unit is a list which references the appropriate cells
The usage of sets maximizes speed for operation like checking if an value is present in the domain, removing elements from the domain itself and checking if the domain is an empty set (such case correspond to an invalid state)
I found that the usage of this approach has several advances, few of which are:
1. When the domain of one cell is shrunk by either a constraint or an assignment, all units which contain the reference to the same cell are automatically updated maintaining the consistency of the same cell domain across all units.
2. Having a ready to use list of units proves advantageous when enforcing the constraints, eliminating the need to form on the fly the list of cell domains included in each unit.



## Result comparison ##

This section aims to compare Backtracking depth-first search performance when using different heuristics to estimate the next best move.

The table below compares the average number of explored nodes for heuristic A and B when the cell value ordering is chosen at random.
Due to the randomness introduce by the choice of the priority on the value assignment, to have a more fair comparison each approach is tested on all 15 Sudokus for 10 times. Each run reports the sum of all explored nodes to solve the 15 Sudokus.

| Sudoku number (hard category) | Explored Nodes Approach A | Explored Nodes  Approach B |
| ------ | ---- | ---|
| Run 1 (Sudoku 1-15) |  64 | 460 |		
| Run 2 (Sudoku 1-15) |  72 | 303 |
| Run 3 (Sudoku 1-15) |  66 | 308 |
| Run 4 (Sudoku 1-15) |  55 | 433 |
| Run 5 (Sudoku 1-15) |  65 | 318 |
| Run 6 (Sudoku 1-15) |  67 | 233 |
| Run 7 (Sudoku 1-15) |  66 | 412 |
| Run 8 (Sudoku 1-15) |  72 | 308 |
| Run 9 (Sudoku 1-15) |  69 | 364 |
| Run 10 (Sudoku 1-15) |  72 | 430 |
| Avarage number of EXPLORED NODES | 66.8 | 355.9 |
| Median EXPLORED NODES | 66.5 | 336 |

**Approach A:** The unit with the least number of values in all the cells domain is selected. For the given unit, the cell domain with the least number of free values (>1) is selected.

**Approach B:** The unit with the most number of values in all the cells domain is selected. For the given unit, the cell with the highest number of free values) in its domain is selected
For both approaches the ordering of the cell value assignment is chosen at random.




Having established that the Approach A performs generally better on the test set, the table below compares the performances of the heuristic used in approach A but with two different ordering approached for the cell values.

| Sudoku number (hard category) | Explored Nodes Approach A - 1 | Explored Nodes  Approach A-2 |
| ------ | ---- | ---|
| Sudoku 1 |  1 | 1 |		
| Sudoku 2 |  7 | 7 |
| Sudoku 3 |  6 | 8 |
| Sudoku 4 |  2 | 9 |
| Sudoku 5 |  1 | 1 |
| Sudoku 6 |  3 | 2 |
| Sudoku 7 |  1 | 1 |
| Sudoku 8 |  12 | 21 |
| Sudoku 9 |  8 | 4 |
| Sudoku 10 |  9 | 6 |
| Sudoku 11 |  2 | 3 |
| Sudoku 13 |  2 | 3 |
| Sudoku 14 |  3 | 1 |
| Sudoku 15 |  1 | 1 |
| TOTAL EXPLORED NODES | 59 | 71 |


**Approach A-1**: Values that are already assigned the most on the whole the grid are selected first.

**Approach A-2**: Values that are assigned the least on the whole the grid are selected first.








## References ##

[1] Russell, S, & Norvig, P 2016, Artificial Intelligence: a Modern Approach, EBook, Global Edition : A Modern Approach, Pearson Education, Limited, Harlow. Available from: ProQuest Ebook Central. [19 September 2021].
