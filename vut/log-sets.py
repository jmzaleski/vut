# https://chatgpt.com/share/6716aa7d-3f08-8010-b702-29eaad8cc714


import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value

# Read data from CSV files
boards_df = pd.read_csv('boards.csv')
log_sets_df = pd.read_csv('log_sets.csv')

# Extract board sizes and demand
board_lengths = boards_df['Board_Length'].tolist()
board_demand = boards_df['Demand'].tolist()

# Extract log sets (grouping by log set)
log_sets = log_sets_df.groupby('Set')

# LP problem to maximize the number of boards cut
prob = LpProblem("Cutting_Stock_with_Log_Selection", LpMaximize)

# Decision variables for choosing a log set (binary variable: 0 or 1 for whether to use the set)
choose_log_set = {log_set: LpVariable(f"use_log_set_{log_set}", cat="Binary") for log_set in log_sets.groups.keys()}

# Decision variables for cutting boards from logs within the chosen log set
cut_vars = {}
for log_set, log_data in log_sets:
    for i, board_len in enumerate(board_lengths):
        for j, (log_len, quantity) in log_data[['Log_Length', 'Quantity']].iterrows():
            cut_vars[(log_set, i, j)] = LpVariable(f"cut_board_{i}_from_log_{log_set}_{log_len}", lowBound=0, cat="Integer")

# Objective: Maximize the total number of boards cut
prob += lpSum(cut_vars[log_set, i, j] for log_set in log_sets.groups.keys() 
                                      for i in range(len(board_lengths)) 
                                      for j in range(len(log_sets.get_group(log_set))))

# Constraints:
# 1. For each log set, ensure total cuts do not exceed the available log quantity
for log_set, log_data in log_sets:
    for j, row in log_data.iterrows():
        log_len = row['Log_Length']
        quantity = row['Quantity']
        prob += lpSum(cut_vars[(log_set, i, j)] * board_lengths[i] 
                      for i in range(len(board_lengths))) <= log_len * quantity * choose_log_set[log_set], f"Log_Length_Constraint_Set_{log_set}_Log_{log_len}"

# 2. Meet the demand for each board size
for i in range(len(board_lengths)):
    prob += lpSum(cut_vars[(log_set, i, j)] 
                  for log_set in log_sets.groups.keys() 
                  for j in range(len(log_sets.get_group(log_set)))) >= board_demand[i], f"Demand_Constraint_Board_{i}"

# 3. Only one log set can be chosen
prob += lpSum(choose_log_set[log_set] for log_set in log_sets.groups.keys()) == 1, "One_Log_Set_Selection_Constraint"

# Solve the problem
prob.solve()

# Output the results
print("Optimal log set selection and cutting plan:")
for log_set in log_sets.groups.keys():
    if value(choose_log_set[log_set]) == 1:
        print(f"Log Set {log_set} selected")
        for i in range(len(board_lengths)):
            for j in range(len(log_sets.get_group(log_set))):
                cut_count = value(cut_vars[(log_set, i, j)])
                if cut_count > 0:
                    log_len = log_sets.get_group(log_set).iloc[j]['Log_Length']
                    print(f"  Cut {cut_count} boards of length {board_lengths[i]} from logs of length {log_len}")

# Total number of boards cut
total_boards_cut = sum(value(cut_vars[(log_set, i, j)]) for log_set in log_sets.groups.keys()
                                                       for i in range(len(board_lengths))
                                                       for j in range(len(log_sets.get_group(log_set))))
print(f"Total boards cut: {total_boards_cut}")
