import streamlit as st
import pandas as pd
import numpy as np
import random
import collections
import copy
import time
import os

DEPOT_ID = 'D0'

def merge_products(dict1, dict2):
    """Merges two dictionaries of products and their quantities."""
    merged = collections.defaultdict(int)
    for p, q in dict1.items():
        merged[p] += q
    for p, q in dict2.items():
        merged[p] += q
    return dict(merged)

def simulate_route_capacity(route, capacity):
    """
    Simulates a route to check if vehicle capacity is exceeded at any point.
    Returns True if capacity is respected, False otherwise.
    Also returns the maximum load achieved during the simulation.
    """
    current_load_dict = collections.defaultdict(int)
    max_load_achieved = 0
    for action in route:
        if action["type"] == "pickup":
            for product, qty in action["products"].items():
                current_load_dict[product] += qty
        elif action["type"] == "delivery":
            # In user's logic, quantity_needed is negative for delivery
            for product, qty in action["products"].items():
                current_load_dict[product] += qty
                if current_load_dict[product] <= 0:
                    del current_load_dict[product]
        
        current_total_load = sum(current_load_dict.values())
        if current_total_load > max_load_achieved:
            max_load_achieved = current_total_load
        if current_total_load > capacity:
            return False, max_load_achieved
    return True, max_load_achieved

def check_and_assign_to_existing_vehicle(vehicle, customer_order, capacity):
    """
    Tries to assign a customer_order to an existing vehicle.
    Returns the MODIFIED vehicle object if successful, or None if it fails.
    This version creates a copy and returns it, not modifying in place.
    """
    # Create a deep copy to avoid modifying the original vehicle if assignment fails
    temp_vehicle = copy.deepcopy(vehicle)

    customer_id = customer_order["CustomerID"]
    sourcing_details = customer_order["sourcing_details"]
    quantity_needed = customer_order["quantity_needed"]

    customer_pickup_operations_info = [] 
    for store_id, products_for_cust_from_store in sourcing_details.items():
        is_merge = store_id in temp_vehicle["visited_stores_for_pickup"]
        current_vehicle_pickup_at_store = temp_vehicle["visited_stores_for_pickup"].get(store_id, {})
        combined_pickup_at_store = merge_products(current_vehicle_pickup_at_store, products_for_cust_from_store)
        
        # Check if merging pickups at a single store stop overloads the vehicle
        if is_merge and sum(combined_pickup_at_store.values()) > capacity:
             return None

        customer_pickup_operations_info.append({
            "store_id": store_id,
            "products_for_this_customer": products_for_cust_from_store, 
            "is_merge": is_merge,
            "final_pickup_at_store_if_assigned": combined_pickup_at_store 
        })

    # Build the proposed new route
    proposed_route = []
    # Part 1: Process existing route, updating merged pickups
    for action in temp_vehicle["route"]:
        action_copy = copy.deepcopy(action)
        if action_copy["type"] == "pickup":
            store_id_in_action = action_copy["store"]
            merge_info = next((op_info for op_info in customer_pickup_operations_info 
                               if op_info["store_id"] == store_id_in_action and op_info["is_merge"]), None)
            if merge_info:
                action_copy["products"] = merge_info["final_pickup_at_store_if_assigned"]
        proposed_route.append(action_copy)

    # Part 2: Add new pickup actions for this customer
    for op_info in customer_pickup_operations_info:
        if not op_info["is_merge"]:
            proposed_route.append({
                "type": "pickup", "store": op_info["store_id"], 
                "products": op_info["products_for_this_customer"]
            })
            
    # Part 3: Add delivery action
    proposed_route.append({
        "type": "delivery", "customer": customer_id, "products": quantity_needed.copy()
    })

    # Simulate the proposed route for overall capacity
    capacity_ok, _ = simulate_route_capacity(proposed_route, capacity)
    if not capacity_ok:
        return None

    # If all checks pass, commit changes to the temporary vehicle object
    temp_vehicle["route"] = proposed_route
    temp_vehicle["assigned_customers"].append(customer_id)
    
    new_visited_stores_summary = collections.defaultdict(lambda: collections.defaultdict(int))
    for action in temp_vehicle["route"]:
        if action["type"] == "pickup":
            store_id = action["store"]
            new_visited_stores_summary[store_id] = action["products"].copy()
    temp_vehicle["visited_stores_for_pickup"] = {k: dict(v) for k, v in new_visited_stores_summary.items()}
    
    return temp_vehicle

def create_new_vehicle_for_customer(customer_order, capacity, vehicle_id):
    customer_id = customer_order["CustomerID"]
    sourcing_details = customer_order["sourcing_details"]
    quantity_needed = customer_order["quantity_needed"]
    new_vehicle_route = []

    for store_id, products_to_pickup in sourcing_details.items():
        if sum(products_to_pickup.values()) > capacity:
            # This individual customer order is too large for an empty vehicle
            return None 
        new_vehicle_route.append({
            "type": "pickup", "store": store_id, "products": products_to_pickup.copy()
        })
    new_vehicle_route.append({
        "type": "delivery", "customer": customer_id, "products": quantity_needed.copy()
    })

    capacity_ok, _ = simulate_route_capacity(new_vehicle_route, capacity)
    if not capacity_ok:
        return None

    visited_stores_summary = collections.defaultdict(lambda: collections.defaultdict(int))
    for action in new_vehicle_route:
        if action["type"] == "pickup":
            visited_stores_summary[action["store"]] = action["products"].copy()

    return {
        "id": vehicle_id, "route": new_vehicle_route, 
        "assigned_customers": [customer_id],
        "visited_stores_for_pickup": {k: dict(v) for k, v in visited_stores_summary.items()}
    }

def solve_vrpdp_custom_chromosome_generation(all_customer_orders, vehicle_capacity):
    """ Main solver function from user, used to create one chromosome. """
    vehicles = []
    vehicle_id_counter = 0
    # The order of customer_orders matters here, so it should be shuffled before calling this function
    unassigned_orders = list(all_customer_orders)

    while unassigned_orders:
        customer_order = unassigned_orders.pop(0)
        customer_id = customer_order["CustomerID"]
        assigned_to_existing = False
        
        # Try to assign to an existing vehicle
        for i, vehicle in enumerate(vehicles):
            # Pass a copy of the vehicle to check_and_assign
            # If it returns a modified vehicle, it was successful
            updated_vehicle = check_and_assign_to_existing_vehicle(vehicle, customer_order, vehicle_capacity)
            if updated_vehicle:
                vehicles[i] = updated_vehicle # Replace the old vehicle with the updated one
                assigned_to_existing = True
                break
        
        if not assigned_to_existing:
            vehicle_id_counter += 1
            new_vehicle_id_str = f"V{vehicle_id_counter}"
            new_vehicle = create_new_vehicle_for_customer(customer_order, vehicle_capacity, new_vehicle_id_str)
            if new_vehicle:
                vehicles.append(new_vehicle)
            else:
                st.warning(f"Could not assign customer {customer_id} to any vehicle or create a new one. This customer will be excluded from this solution.")

    return tuple(vehicles)

def create_sourcing_plans(demands_df, initial_stocks_df):
    """ Generates sourcing plans (All_Ts) for all customers. """
    max_attempts = 15
    for attempt in range(max_attempts):
        current_stocks_df = initial_stocks_df.copy()
        all_T = []
        customer_ids = list(demands_df['CustomerID'].unique())
        random.shuffle(customer_ids)
        all_sourced_successfully = True
        for customer_id in customer_ids:
            customer_demands = demands_df[demands_df['CustomerID'] == customer_id]
            t_plan = {"CustomerID": customer_id, "sourcing_details": {}, "quantity_needed": {}}
            sourcing_failed = False
            for _, demand in customer_demands.iterrows():
                prod_id, qty_needed = demand["ProductID"], demand["Quantity"]
                t_plan["quantity_needed"][prod_id] = t_plan["quantity_needed"].get(prod_id, 0) - qty_needed
                sourced_for_this_product = 0
                stores_df = current_stocks_df[(current_stocks_df["ProductID"] == prod_id) & (current_stocks_df["StockQuantity"] > 0)].copy()
                if stores_df.empty and qty_needed > 0: sourcing_failed = True; break
                store_indices = list(stores_df.index); random.shuffle(store_indices)
                for store_idx in store_indices:
                    if sourced_for_this_product >= qty_needed: break
                    store_id = stores_df.loc[store_idx, "StoreID"]
                    stock_idx = current_stocks_df[(current_stocks_df["StoreID"] == store_id) & (current_stocks_df["ProductID"] == prod_id)].index[0]
                    stock_in_store = current_stocks_df.loc[stock_idx, "StockQuantity"]
                    qty_to_take = min(qty_needed - sourced_for_this_product, stock_in_store)
                    if qty_to_take > 0:
                        if store_id not in t_plan["sourcing_details"]: t_plan["sourcing_details"][store_id] = {}
                        t_plan["sourcing_details"][store_id][prod_id] = t_plan["sourcing_details"][store_id].get(prod_id, 0) + qty_to_take
                        sourced_for_this_product += qty_to_take
                        current_stocks_df.loc[stock_idx, "StockQuantity"] -= qty_to_take
                if sourced_for_this_product < qty_needed: sourcing_failed = True; break
            if sourcing_failed: all_sourced_successfully = False; break
            all_T.append(t_plan)
        if all_sourced_successfully: return all_T
    return None

# --- Genetic Algorithm Components ---
def get_location_sequence_from_actions(route_actions, depot_id_val):
    location_sequence = [depot_id_val]
    for action in route_actions:
        if action["type"] == "pickup":
            location_sequence.append(action["store"])
        elif action["type"] == "delivery":
            location_sequence.append(action["customer"])
    location_sequence.append(depot_id_val)
    if not location_sequence: return []
    final_sequence = [location_sequence[0]]
    for i in range(1, len(location_sequence)):
        if location_sequence[i] != final_sequence[-1]:
            final_sequence.append(location_sequence[i])
    return final_sequence

def calculate_chromosome_fitness(chromosome, distance_matrix, vehicle_cost=100):
    total_distance = 0
    if not chromosome: return float('inf')
    num_vehicles = len(chromosome)
    for vehicle_data in chromosome:
        location_sequence = get_location_sequence_from_actions(vehicle_data['route'], DEPOT_ID)
        for i in range(len(location_sequence) - 1):
            loc1, loc2 = location_sequence[i], location_sequence[i+1]
            dist = distance_matrix.get(loc1, {}).get(loc2, 9999)
            if pd.isna(dist): dist = 9999
            total_distance += dist
    return total_distance + (num_vehicles * vehicle_cost)

def selection_tournament(population, fitness_scores, tournament_size=3):
    selected_parents = []
    population_with_fitness = list(zip(population, fitness_scores))
    for _ in range(len(population)):
        tournament_contenders = random.sample(population_with_fitness, tournament_size)
        winner = min(tournament_contenders, key=lambda x: x[1])
        selected_parents.append(winner[0])
    return selected_parents

def crossover(parent1, parent2, all_T_plans_global, vehicle_capacity):
    all_customer_ids = list(set(t['CustomerID'] for t in all_T_plans_global))
    if len(all_customer_ids) < 2: return copy.deepcopy(parent1), copy.deepcopy(parent2)
    random.shuffle(all_customer_ids)
    split_point = random.randint(1, len(all_customer_ids) - 1)
    child1_cust_set = set(all_customer_ids[:split_point])
    child1_T = [t for t in all_T_plans_global if t['CustomerID'] in child1_cust_set]
    child2_T = [t for t in all_T_plans_global if t['CustomerID'] not in child1_cust_set]
    child1 = solve_vrpdp_custom_chromosome_generation(child1_T + child2_T, vehicle_capacity)
    child2 = solve_vrpdp_custom_chromosome_generation(child2_T + child1_T, vehicle_capacity)
    return child1, child2

def mutation(chromosome, mutation_rate=0.2):
    mutated_chromosome = list(copy.deepcopy(chromosome))
    if not mutated_chromosome or random.random() > mutation_rate: return tuple(mutated_chromosome)
    vehicle_to_mutate_idx = random.randrange(len(mutated_chromosome))
    route = mutated_chromosome[vehicle_to_mutate_idx]["route"]
    if len(route) >= 2:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return tuple(mutated_chromosome)

def run_genetic_algorithm(all_T_plans, vehicle_capacity, distance_matrix, pop_size, generations, progress_bar, status_text, timer_placeholder):
    start_time = time.time()
    status_text.text("Step 1/3: Initializing population...")
    population = []
    for i in range(pop_size):
        shuffled_T = list(all_T_plans); random.shuffle(shuffled_T)
        individual = solve_vrpdp_custom_chromosome_generation(shuffled_T, vehicle_capacity)
        if individual: population.append(individual)
        progress_bar.progress(i / pop_size)
        elapsed_time = time.time() - start_time
        timer_placeholder.metric("Elapsed Time", f"{elapsed_time:.2f} s")
        
    if not population:
        status_text.error("Could not initialize population."); return None, None, 0
    best_chromosome, best_fitness = None, float('inf')
    
    status_text.text("Step 2/3: Running generations...")
    for gen in range(generations):
        fitness_scores = [calculate_chromosome_fitness(ind, distance_matrix) for ind in population]
        best_current_idx = np.argmin(fitness_scores)
        if fitness_scores[best_current_idx] < best_fitness:
            best_fitness, best_chromosome = fitness_scores[best_current_idx], copy.deepcopy(population[best_current_idx])
        
        elapsed_time = time.time() - start_time
        status_text.text(f"Generation {gen + 1}/{generations} | Best Fitness: {best_fitness:.2f}")
        timer_placeholder.metric("Elapsed Time", f"{elapsed_time:.2f} s")
        progress_bar.progress((gen + 1) / generations)

        selected = selection_tournament(population, fitness_scores)
        next_pop = [copy.deepcopy(best_chromosome)]
        while len(next_pop) < pop_size:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2, all_T_plans, vehicle_capacity)
            c1, c2 = mutation(c1), mutation(c2)
            if c1: next_pop.append(c1)
            if c2 and len(next_pop) < pop_size: next_pop.append(c2)
        population = next_pop
        
    status_text.text("Step 3/3: GA Finished!")
    progress_bar.progress(1.0)
    final_duration = time.time() - start_time
    timer_placeholder.metric("Elapsed Time", f"{final_duration:.2f} s")
    time.sleep(1)
    return best_chromosome, best_fitness, final_duration

def create_product_flow_matrix(vehicle_route_actions, all_product_ids):
    location_sequence = []
    for action in vehicle_route_actions:
        location_sequence.append(action.get('store') or action.get('customer'))
    if not location_sequence:
        return pd.DataFrame(index=all_product_ids)
        
    flow_df = pd.DataFrame(0, index=all_product_ids, columns=location_sequence, dtype=int)
    for i, action in enumerate(vehicle_route_actions):
        col_name = action.get('store') or action.get('customer')
        for prod, qty in action['products'].items():
            if prod in flow_df.index:
                flow_df.loc[prod, col_name] = flow_df.loc[prod, col_name] + qty if action['type'] == 'pickup' else qty
    return flow_df

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🚚 VRP-SPD Genetic Algorithm")

with st.sidebar:
    st.header("⚙️ Configuration")
    demands_file = st.file_uploader("1. Upload Demands CSV", type="csv")
    stocks_file = st.file_uploader("2. Upload Stocks CSV", type="csv")
    distances_file = st.file_uploader("3. Upload Distances CSV", type="csv")
    st.markdown("---")
    vehicle_capacity_input = st.number_input("Vehicle Capacity", min_value=1, value=20)
    population_size_input = st.number_input("Population Size", min_value=10, max_value=200, value=50, step=10)
    generations_input = st.number_input("Number of Generations", min_value=1, max_value=500, value=100, step=10)
    run_button = st.button("🚀 Run Genetic Algorithm")

if run_button:
    if demands_file and stocks_file and distances_file:
        try:
            demands_df = pd.read_csv(demands_file); stocks_df = pd.read_csv(stocks_file)
            dist_matrix = pd.read_csv(distances_file, index_col=0).to_dict('index')
            all_prod_ids = sorted(list(demands_df['ProductID'].unique()))
            
            st.info("Data loaded. Starting GA process...")
            
            with st.spinner("Step 1: Creating sourcing plans..."):
                all_T_plans = create_sourcing_plans(demands_df, stocks_df)
            if not all_T_plans:
                st.error("Sourcing failed. Check if stock is sufficient for all demands.")
            else:
                st.success(f"Sourcing successful for {len(all_T_plans)} customers.")

                # --- UI Placeholders for real-time updates ---
                status_text = st.empty()
                progress_bar = st.progress(0)
                timer_placeholder = st.empty()
                
                best_solution, best_fitness, total_duration = run_genetic_algorithm(
                    all_T_plans, vehicle_capacity_input, dist_matrix,
                    population_size_input, generations_input, 
                    progress_bar, status_text, timer_placeholder
                )
                
                # Clear progress elements and show final results
                status_text.empty(); progress_bar.empty(); timer_placeholder.empty()

                if best_solution:
                    st.header("🏆 Best Solution Found")
                    pure_distance = calculate_chromosome_fitness(best_solution, dist_matrix, vehicle_cost=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Cost (Fitness)", f"{best_fitness:.2f}")
                    col2.metric("Pure Distance", f"{pure_distance:.2f}")
                    col3.metric("Vehicles Used", len(best_solution))
                    col4.metric("Total Duration", f"{total_duration:.2f} s")

                    for i, vehicle_route in enumerate(best_solution):
                        with st.expander(f"🚛 Vehicle {i+1} ({vehicle_route['id']}) - Route & Details", expanded=i==0):
                            st.subheader("Route Sequence")
                            location_seq = get_location_sequence_from_actions(vehicle_route['route'], DEPOT_ID)
                            st.code(" -> ".join(location_seq))
                            st.subheader("Product Flow Matrix")
                            flow_matrix = create_product_flow_matrix(vehicle_route['route'], all_prod_ids)
                            st.dataframe(flow_matrix)
                else: st.error("GA did not produce a valid solution.")
        except Exception as e: st.error(f"An error occurred: {e}"); st.exception(e)
    else: st.warning("Please upload all three required CSV files.")
else: st.info("Upload your data and configure parameters in the sidebar, then click 'Run'.")
