# https://chatgpt.com/share/6716aa7d-3f08-8010-b702-29eaad8cc714

import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value
import math

# Read data from CSV files
boards_df = pd.read_csv('boards.csv')
log_sets_df = pd.read_csv('log_sets.csv')

# Extract board sizes (length, width, height) and demand
board_lengths = boards_df['Board_Length'].tolist()
board_widths = boards_df['Board_Width'].tolist()
board_heights = boards_df['Board_Height'].tolist()
board_demand = boards_df['Demand'].tolist()

# Extract log sets (grouping by log set)
log_sets = log_sets_df.groupby('Set')

# LP problem to minimize wastage
prob = LpProblem("Cutting_Stock_with_Minimal_Wastage", LpMinimize)

# Decision variables for choosing a log set (binary variable: 0 or 1 for whether to use the set)
choose_log_set = {log_set: LpVariable(f"use_log_set_{log_set}", cat="Binary") for log_set in log_sets.groups.keys()}

# Decision variables for cutting boards from logs within the chosen log set
cut_vars = {}
for log_set, log_data in log_sets:
    for i, board_len in enumerate(board_lengths):
        for j, (log_len, log_diam, quantity) in log_data[['Log_Length', 'Log_Diameter', 'Quantity']].iterrows():
            cut_vars[(log_set, i, j)] = LpVariable(f"cut_board_{i}_from_log_{log_set}_{log_len}", lowBound=0, cat="Integer")

# Objective: Minimize the total wastage (difference between log volume and used board volume)
wastage_terms = []
for log_set, log_data in log_sets:
    for j, row in log_data.iterrows():
        log_len = row['Log_Length']
        log_diam = row['Log_Diameter']
        quantity = row['Quantity']
        
        # Log volume (cylinder) = pi * (D/2)^2 * L
        log_volume = math.pi * (log_diam / 2) ** 2 * log_len * quantity
        
        # Total volume of boards cut from this log set
        board_volume_cut = lpSum(cut_vars[(log_set, i, j)] * board_lengths[i] * board_widths[i] * board_heights[i] 
                                 for i in range(len(board_lengths)))
        
        # Wastage is the difference between log volume and board volume cut
        wastage = log_volume - board_volume_cut
        wastage_terms.append(wastage * choose_log_set[log_set])

# Set the objective to minimize wastage
prob += lpSum(wastage_terms), "Total_Wastage"

# Constraints:
# 1. For each log set, ensure total cuts do not exceed the available log length and volume
for log_set, log_data in log_sets:
    for j, row in log_data.iterrows():
        log_len = row['Log_Length']
        log_diam = row['Log_Diameter']
        quantity = row['Quantity']
        
        # Log volume (cylinder) = pi * (D/2)^2 * L
        log_volume = math.pi * (log_diam / 2) ** 2 * log_len
        
        # Length constraint: Do not exceed the log's length
        prob += lpSum(cut_vars[(log_set, i, j)] * board_lengths[i] 
                      for i in range(len(board_lengths))) <= log_len * quantity * choose_log_set[log_set], f"Length_Constraint_Set_{log_set}_Log_{log_len}"

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
        total_wastage = 0
        for i in range(len(board_lengths)):
            for j in range(len(log_sets.get_group(log_set))):
                cut_count = value(cut_vars[(log_set, i, j)])
                if cut_count > 0:
                    log_len = log_sets.get_group(log_set).iloc[j]['Log_Length']
                    log_diam = log_sets.get_group(log_set).iloc[j]['Log_Diameter']
                    print(f"  Cut {cut_count} boards of length {board_lengths[i]}, width {board_widths[i]}, height {board_heights[i]} from logs of length {log_len} and diameter {log_diam}")
                    
                    # Calculate the total wastage for this log set
                    log_volume = math.pi * (log_diam / 2) ** 2 * log_len
                    board_volume_cut = cut_count * board_lengths[i] * board_widths[i] * board_heights[i]
                    wastage = log_volume - board_volume_cut
                    total_wastage += wastage
        
        print(f"Total wastage for Log Set {log_set}: {total_wastage:.2f}")

# Output total wastage
total_wastage = sum(value(wastage_term) for wastage_term in wastage_terms)
print(f"Total wastage: {total_wastage:.2f}")
