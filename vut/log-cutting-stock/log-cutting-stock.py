
# Explanation:
# Input Files: The data for the boards (boards.csv) and logs (logs.csv) is read using pandas.
# Decision Variables: For each combination of board length and log, we create a decision variable cut_vars[i, j], which represents how many times a board of size i will be cut from log j.
# Objective Function: The goal is to maximize the total number of boards cut, which is represented by the sum of all decision variables.
# Constraints:
# Log Constraints: Ensure that the total length of boards cut from a log doesn’t exceed the log’s length.
# Demand Constraints: Ensure that at least the required number of each board size is produced.
# Solve the Problem: The solve() function from PuLP solves the linear programming model.

import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value

# Read data from CSV files
boards_df = pd.read_csv('boards.csv')
logs_df = pd.read_csv('logs.csv')

# Extract board sizes and demand
board_lengths = boards_df['Board_Length'].tolist()
board_demand = boards_df['Demand'].tolist()

# Extract log lengths
log_lengths = logs_df['Log_Length'].tolist()

# Create LP problem
prob = LpProblem("Cutting_Stock_Problem", LpMaximize)

# Decision variables: How many times to cut each board length from each log
cut_vars = {(i, j): LpVariable(f"cut_board_{i}_from_log_{j}", lowBound=0, cat="Integer")
            for i in range(len(board_lengths)) 
            for j in range(len(log_lengths))}

# Objective: Maximize the total number of boards cut
prob += lpSum(cut_vars[i, j] for i in range(len(board_lengths)) for j in range(len(log_lengths)))

# Constraints:
# 1. Do not exceed the log length when cutting boards
for j in range(len(log_lengths)):
    prob += lpSum(cut_vars[i, j] * board_lengths[i] for i in range(len(board_lengths))) <= log_lengths[j], f"Log_Constraint_{j}"

# 2. Meet the demand for each board size
for i in range(len(board_lengths)):
    prob += lpSum(cut_vars[i, j] for j in range(len(log_lengths))) >= board_demand[i], f"Demand_Constraint_{i}"

# Solve the problem
prob.solve()

# Output the results
print("Optimal cutting plan:")
for i in range(len(board_lengths)):
    for j in range(len(log_lengths)):
        cut_count = value(cut_vars[i, j])
        if cut_count > 0:
            print(f"Cut {cut_count} boards of length {board_lengths[i]} from log {log_lengths[j]}")

# Total number of boards cut
total_boards_cut = sum(value(cut_vars[i, j]) for i in range(len(board_lengths)) for j in range(len(log_lengths)))
print(f"Total boards cut: {total_boards_cut}")
