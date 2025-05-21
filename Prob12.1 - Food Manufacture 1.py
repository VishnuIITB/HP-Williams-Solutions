""" This is HP Williams problem no 12.1"""
import pulp
from pulp import lpSum, LpVariable, LpProblem, LpConstraint, LpMaximize, LpBinary, LpInteger, LpContinuous
import pandas as pd
import csv

# Data
df_price = pd.read_csv(f'C:\\Users\\VishnuJ\\Desktop\\Knwoledge Lab\\Operations Research Laboratory\\HP Williams Solutions\\Prob12.1 - Food Manufacture 1\\Data 12.1\\Oil_Prices.csv', index_col=0)  #Buying price of oil i in month j
# print(df_price)
df_hardness = pd.read_csv(f'C:\\Users\\VishnuJ\\Desktop\\Knwoledge Lab\\Operations Research Laboratory\\HP Williams Solutions\\Prob12.1 - Food Manufacture 1\\Data 12.1\\Hardness_Values.csv', index_col=0)               #Hardness of oil type i
# print(df_hardness.head())


oils_list = ['VEG1', 'VEG2', 'OIL1', 'OIL2', 'OIL3']       # set of oils
month_list= ['January', 'February', 'March', 'April', 'May', 'June']        #set of months

price_per_unit_of_oil_in_month = {}
for oil in oils_list:
    for month in month_list:
        price_per_unit_of_oil_in_month[oil, month] = df_price.loc[month, oil] #  because months are rows and oils are columns in df_price dataframe

hardness_of_oil = {}
for oil in oils_list:
    hardness_of_oil[oil] = df_hardness.loc[oil, 'Hardness Value']

selling_price_per_unit = 150         # Selling price of final product
inventory_cost_per_unit_per_month = 5       # Storage cost per ton per month
monthly_refining_capacity_veg = 200    # Monthly refining capacity of veg oils
monthly_refining_capacity_nonveg = 250     # Monthly refining capacity of non-veg oils
monthly_inventory_holding_capacity_for_oil= 1000       # Inventory holding capacity for raw oils
initial_inventory_of_oil = {i:500 for i in oils_list}        # Initial inventory available at the beginning of the January
final_inventory_of_oil = 500           # Final inventory needed at the end of the June
max_hardness = 6               # Upper bound of hardness
min_hardness = 3               # Lower bound of hardness
#
#
# Decision Variables
qty_purchased_of_oil_in_month = {}  # x_ij represents the oil i quantity purchased in month j
for oil in oils_list:
    for month in month_list:
        qty_purchased_of_oil_in_month[oil,month] = pulp.LpVariable(f'qty_purchased_of_oil_{oil}_in_month_{month}', lowBound=0, cat= LpContinuous)

qty_refined_of_oil_in_month = {}   # y_ij represents the oil quantity refined in month j
for oil in oils_list:
    for month in month_list:
        qty_refined_of_oil_in_month[oil,month] = pulp.LpVariable(f'qty_refined_of_oil_{oil}_in_month_{month}', lowBound=0, cat=LpContinuous)

qty_stored_of_oil_at_month_end = {}   # y_ij represents the inventory of oil i at the end of month j
for oil in oils_list:
    for month in month_list:
        qty_stored_of_oil_at_month_end[oil,month] = pulp.LpVariable(f'qty_stored_of_oil_{oil}_at_month_end_{month}', lowBound=0, cat=LpContinuous)

#************************************************************************  Objective Function ***************************************************************************************
# Objective Function
model = pulp.LpProblem('Food Manufacture 1', sense = pulp.LpMaximize)

object_expr = 0
for oil in oils_list:
    for month in month_list:
        object_expr += (selling_price_per_unit * qty_refined_of_oil_in_month[oil,month] - price_per_unit_of_oil_in_month[oil,month] * qty_purchased_of_oil_in_month[oil,month] - inventory_cost_per_unit_per_month*qty_stored_of_oil_at_month_end[oil,month])
model += object_expr

# Count number of terms in the objective function
objective_terms = model.objective
term_count = len(objective_terms.to_dict())
print('Objective Function:', objective_terms)
print(f"Number of terms in the objective function: {term_count}")


#***************************************************************************  Constraints 1  ********************************************************************************************
# Constraints 1 -  Inventory balancing constraint

# Inventory Balancing for January
for oil in oils_list:
        model += initial_inventory_of_oil[oil] + qty_purchased_of_oil_in_month[oil,'January'] == qty_refined_of_oil_in_month[oil,'January'] + qty_stored_of_oil_at_month_end[oil,'January'], f'Inventory_Balance_Constraint_for_{oil}_January'   # Inventory Balancing for January
model.writeLP("Food_Manufacture_Model.lp")
# printing Inventory balancing constraint for January
print("\nInventory Balance Constraints for January:")
count_constraints = 0
for oil in oils_list:
    cname = f"Inventory_Balance_Constraint_for_{oil}_January"
    constraint_expr = model.constraints[cname]
    constraint_str = str(constraint_expr)
    num_terms = len(constraint_expr.toDict())
    print(f"{cname}: {constraint_str} | Length of expression (chars): {len(constraint_str)} | Number of terms: {num_terms}")
    count_constraints += 1
print(f"\nTotal number of 'Inventory Balance' constraints for January: {count_constraints}")

# Inventory Balancing from February to June
for oil in oils_list:
    for month in range(1,len(month_list)):
        current_month = month_list[month]
        prev_month = month_list[month-1]
        model += qty_stored_of_oil_at_month_end[oil,prev_month] + qty_purchased_of_oil_in_month[oil,current_month] == qty_refined_of_oil_in_month[oil,current_month] + qty_stored_of_oil_at_month_end[oil, current_month], f'Inventory_Balance_Constraint_for_{oil}_{month}'

# printing Inventory balancing constraint from February to June
print("\nInventory Balance Constraints (Feb to June):")
count_constraints = 0
for oil in oils_list:
    for month in range(1, len(month_list)):
        current_month = month_list[month]
        cname = f"Inventory_Balance_Constraint_for_{oil}_{month}"
        constraint_expr = model.constraints[cname]
        constraint_str = str(constraint_expr)
        num_terms = len(constraint_expr.toDict())
        print(f"{cname}: {constraint_str} | Length of expression (chars): {len(constraint_str)} | Number of terms: {num_terms}")
        count_constraints += 1
print(f"\nTotal number of 'Inventory Balance' constraints (Feb to June): {count_constraints}")

# Alternative way to write Inventory balancing constraint for Feb to June.
# for i in oils_list:
#     for j in month_list[1:]:
#         model += z[i, month_list[month_list.index(j)-1]] + x[i,j] == y[i,j] + z[i,j], f'Inventory_Balance_Constraint_{i}_{j}'
model.writeLP("Food_Manufacture_Model.lp")

#***************************************************************************   Constraints 2   ********************************************************************************************
# Constraint 2 - Refining capacity constraint for Veg and Nog veg oils
for month in month_list:
    model += lpSum(qty_refined_of_oil_in_month[oil,month] for oil in oils_list[:2]) <= monthly_refining_capacity_veg, f'Veg_refining_capacity_{month}'
    model += lpSum(qty_refined_of_oil_in_month[oil,month] for oil in oils_list[2:]) <= monthly_refining_capacity_nonveg, f'NonVeg_refining_capacity_{month}'

# printing Refining capacity constraint for Veg
print("\nVeg Refining Capacity Constraints:")
count_veg = 0
for month in month_list:
    cname = f'Veg_refining_capacity_{month}'
    constraint_expr = model.constraints[cname]
    constraint_str = str(constraint_expr)
    num_terms = len(constraint_expr.toDict())
    print(f"{cname}: {constraint_str} | Length of expression (chars): {len(constraint_str)} | Number of terms: {num_terms}")
    count_veg += 1
print(f"Total Veg refining capacity constraints: {count_veg}")

# printing Refining capacity constraint for NonVeg
print("\nNonVeg Refining Capacity Constraints:")
count_nonveg = 0
for month in month_list:
    cname = f'NonVeg_refining_capacity_{month}'
    constraint_expr = model.constraints[cname]
    constraint_str = str(constraint_expr)
    num_terms = len(constraint_expr.toDict())
    print(f"{cname}: {constraint_str} | Length of expression (chars): {len(constraint_str)} | Number of terms: {num_terms}")
    count_nonveg += 1
print(f"Total NonVeg refining capacity constraints: {count_nonveg}")

#***************************************************************************   Constraints 3   ********************************************************************************************
 # Constraint 3 - Hardness balancing constraint
for month in month_list:
    model += lpSum(hardness_of_oil[oil] * qty_refined_of_oil_in_month[oil,month] for oil in oils_list) <= max_hardness * lpSum(qty_refined_of_oil_in_month[oil,month] for oil in oils_list), f"Hardness_Upper_{month}"
    model += lpSum(hardness_of_oil[oil] * qty_refined_of_oil_in_month[oil,month] for oil in oils_list) >= min_hardness * lpSum(qty_refined_of_oil_in_month[oil,month] for oil in oils_list), f"Hardness_Lower_{month}"



# # printing Hardness balancing constraint Upper bound
# print("\nHardness Upper Bound Constraints:")
# count_upper = 0
# for month in month_list:
#     cname = f'Hardness_Upper_{month}'
#     constraint_expr = model.constraints[cname]
#     constraint_str = str(constraint_expr)
#     num_terms = len(constraint_expr.toDict())
#     print(f"{cname}: {constraint_str} | Length of expression (chars): {len(constraint_str)} | Number of terms: {num_terms}")
#     count_upper += 1
# print(f"Total Hardness Upper Bound constraints: {count_upper}")
#
# # printing Hardness balancing constraint Lower bound
# print("\nHardness Lower Bound Constraints:")
# count_lower = 0
# for month in month_list:
#     cname = f'Hardness_Lower_{month}'
#     constraint_expr = model.constraints[cname]
#     constraint_str = str(constraint_expr)
#     num_terms = len(constraint_expr.toDict())
#     print(f"{cname}: {constraint_str} | Length of expression (chars): {len(constraint_str)} | Number of terms: {num_terms}")
#     count_lower += 1
# print(f"Total Hardness Lower Bound constraints: {count_lower}")

##***************************************************************************   Constraints 4   ********************************************************************************************
# Constraint 4 - Final Inventory Constraint for June
for oil in oils_list:
    model += qty_stored_of_oil_at_month_end[oil,'June'] == final_inventory_of_oil, f'Inventory_constraint_for_{oil}_June'

# Printing final inventory constraint for June
# Printing final inventory constraint for June
print("\nFinal Inventory Constraints for June:")
count = 0
for oil in oils_list:
    cname = f'Inventory_constraint_for_{oil}_June'
    constraint_expr = model.constraints[cname]
    expr_str = str(constraint_expr)
    length_expr = len(expr_str)
    num_terms = len(constraint_expr.toDict())
    print(f"{cname}: {expr_str} | Length of expression (chars): {length_expr} | Number of terms: {num_terms}")
    count += 1
print(f"Total number of Final Inventory Constraints for June: {count}")

#***************************************************************************   Constraints 5   ********************************************************************************************
# Constraint 5 - Inventory Capacity Constraint
for oil in oils_list:
    for month in month_list:
        model += qty_stored_of_oil_at_month_end[oil,month] <= monthly_inventory_holding_capacity_for_oil, f'Inventory_capacity_constraint_for_{oil}_in_{month}'

# # printing Inventory Capacity Constraint for every month
# print("\nInventory Capacity Constraints:")
# count = 0
# for oil in oils_list:
#     for month in month_list:
#         cname = f'Inventory_capacity_constraint_{oil}_in_{month}'
#         constraint_expr = model.constraints[cname]
#         expr_str = str(constraint_expr)
#         length_expr = len(expr_str)
#         num_terms = len(constraint_expr.toDict())
#         print(f"{cname}: {expr_str} | Length of expression (chars): {length_expr} | Number of terms: {num_terms}")
#         count += 1
# print(f"Total number of Inventory Capacity Constraints: {count}")
#***************************************************************************   Constraints 6   ********************************************************************************************
# Constraint 6 - Mass Conservation Constraint
model += lpSum(qty_purchased_of_oil_in_month for oil in oils_list for month in month_list) == lpSum(qty_refined_of_oil_in_month for oil in oils_list for month in month_list)



#***************************************************************************   Solve Model   ********************************************************************************************
# *********************************************************************************************************
# Solve the model
solver = pulp.PULP_CBC_CMD(msg=True)   # You can also use msg=False to suppress solver output
result_status = model.solve(solver)

# *********************************************************************************************************
# Display Solver Status
print("\nSolver Status:", pulp.LpStatus[model.status])

# *********************************************************************************************************
# Display Objective Value
print("Optimal Profit (Objective Function Value):", pulp.value(model.objective))

# *********************************************************************************************************
# Display Values of Decision Variables
print("\nOptimal Values of Decision Variables:\n")
for var in model.variables():
    if var.varValue > 1e-6:  # Print only non-zero variables
        print(f"{var.name} = {var.varValue:.2f}")

# *********************************************************************************************************
# Optional: Export results to a CSV file
results = []
for var in model.variables():
    if var.varValue > 1e-6:
        results.append({
            'Variable': var.name,
            'Value': var.varValue
        })

df_results = pd.DataFrame(results)
df_results.to_csv("C:\\Users\\VishnuJ\\Desktop\\Food_Manufacture_Solution.csv", index=False)
print("\nResults exported to 'Food_Manufacture_Solution.csv'")

#***************************************************************************   Print Solution   ********************************************************************************************
#Print Solution
def export_solution_table_formatted(oils_list, month_list, x, y, z, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header row
        writer.writerow(["Month", "Buy", "Use/Refine", "Store"])

        for month in month_list:
            # Prepare lists for Buy, Use/Refine, and Store
            buy_list = []
            use_list = []
            store_list = []

            for oil in oils_list:
                buy_qty = x[oil, month].varValue if x[oil, month].varValue is not None else 0
                use_qty = y[oil, month].varValue if y[oil, month].varValue is not None else 0
                store_qty = z[oil, month].varValue if z[oil, month].varValue is not None else 0

                if buy_qty > 0.01:
                    buy_list.append(f"{buy_qty:.1f} tons{oil}")
                if use_qty > 0.01:
                    use_list.append(f"{use_qty:.1f} tons{oil}")
                if store_qty > 0.01:
                    store_list.append(f"{store_qty:.1f} tons{oil}")

            # If no values, show "Nothing"
            if not buy_list: buy_list = ["Nothing"]
            if not use_list: use_list = ["Nothing"]
            if not store_list: store_list = ["Nothing"]

            # Align by rows (zipping columns together)
            max_len = max(len(buy_list), len(use_list), len(store_list))
            for i in range(max_len):
                row = [
                    month if i == 0 else "",  # Only show month in first row of block
                    buy_list[i] if i < len(buy_list) else "",
                    use_list[i] if i < len(use_list) else "",
                    store_list[i] if i < len(store_list) else ""
                ]
                writer.writerow(row)

            writer.writerow([])  # Empty row between months


# Export if solved optimally
if pulp.LpStatus[model.status] == 'Optimal':
    output_csv_path = r"C:\Users\VishnuJ\Desktop\Knwoledge Lab\Operations Research Laboratory\HP Williams Solutions\Prob12.1 - Food Manufacture 1\table12.1_output.csv"
    export_solution_table_formatted(oils_list, month_list, qty_purchased_of_oil_in_month, qty_refined_of_oil_in_month, qty_stored_of_oil_at_month_end, output_csv_path)
    print(f"Excel-formatted table exported to:\n{output_csv_path}")
else:
    print("No optimal solution found.")



