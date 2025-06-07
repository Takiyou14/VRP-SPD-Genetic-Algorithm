# VRPDP Genetic Algorithm Solver with Streamlit

This project provides an interactive web application built with Streamlit to solve a complex variant of the Vehicle Routing Problem with Pickups and Deliveries (VRPDP). It uses a Genetic Algorithm (GA) to find an optimized set of vehicle routes that service all customer demands by picking up products from multiple stores, while respecting vehicle capacity and minimizing total travel distance.

The application allows for interleaved pickup and delivery operations, meaning a vehicle can perform a delivery and then proceed to another pickup within the same tour, reflecting more realistic logistics scenarios.

## Features

-   **Interactive Web Interface:** Built with Streamlit for easy configuration and use.
    
-   **Dynamic Data Upload:** Users can upload their own problem instances via three CSV files: `demands.csv`, `stocks.csv`, and `distances.csv`.
    
-   **Genetic Algorithm Core:** Employs a GA to evolve solutions over multiple generations, aiming for lower total distance and fewer vehicles.
    
-   **Interleaved Routing Logic:** The chromosome generation logic allows for complex routes where pickups and deliveries can be mixed, unlike classic VRP solvers.
    
-   **Dynamic Capacity Simulation:** Vehicle capacity is checked dynamically at every step of a potential route to ensure validity.
    
-   **Configurable Parameters:** Users can easily set the Population Size and Number of Generations for the GA.
    
-   **Real-time Progress:** The interface provides real-time updates on the GA's progress, including elapsed time.
    
-   **Detailed Solution Visualization:** The best solution found is displayed with a clear breakdown per vehicle, including the final route sequence and a detailed **Product Flow Matrix**.
    

## How It Works

The solver is divided into two main phases:

### 1. Sourcing Phase (Generating `All_T`)

Before the GA starts, the application first processes the `demands.csv` and `stocks.csv` files to create a set of feasible "sourcing plans" (referred to as `All_T` in the code).

-   For each customer, it determines which stores can fulfill their specific product demands.
    
-   It randomly assigns products to be picked up from specific stores, ensuring that the store has available stock.
    
-   This process is repeated in multiple attempts if a valid sourcing plan for all customers isn't found on the first try, ensuring a robust start.
    

### 2. Genetic Algorithm Phase

Once a valid set of sourcing plans is established, the GA proceeds to find the best routing solution.

-   **Chromosome Representation:** Each "chromosome" (or individual solution) in the population represents a complete set of vehicle routes that services all customers. It is generated using the user-provided `solve_vrpdp_custom` logic, which intelligently assigns customers to vehicles based on capacity and store visit constraints.
    
-   **Fitness Calculation:** The fitness of each chromosome is calculated based on a weighted sum of the **total distance traveled** (from `distances.csv`) and the **number of vehicles used**. A lower fitness score is better.
    
-   **Evolution:** The algorithm evolves the population over a set number of generations using:
    
    -   **Selection:** A tournament selection method is used to choose parent chromosomes for the next generation.
        
    -   **Crossover:** Parent solutions are combined to create new offspring, mixing customer assignments to explore new routing possibilities.
        
    -   **Mutation:** A small, random change (like swapping two actions in a vehicle's route) is introduced to maintain genetic diversity and avoid premature convergence.
        
-   **Elitism:** The best solution found so far is always carried over to the next generation, ensuring that the quality of the solution never degrades.
    

## File Formats

To use your own data, please format your CSV files as follows.

### `demands.csv`

Contains customer order information.

| CustomerID| ProductID | Quantity |
| :----------------: | :------: | :----: |
| C1       |   P2 | 1 |
| C2           |   P1| 3 |
| C2    |  P4| 1 |
| ...|  ...| ... |

### `stocks.csv`

Contains the available stock for each product at each store.

| StoreID| ProductID | StockQuantity|
| :----------------: | :------: | :----: |
| M1       |   P1 | 88 |
| M1           |   P5| 78 |
| M2    |  P1| 77 |
| ...|  ...| ... |

### `distances.csv`

A matrix containing the distance between any two locations (Depot, Customers, Stores). The first column should be the index. The Depot must be named `D0`.

| Location| D0| C1| C2| M1| M2|...|
| :-----: |:-:|:-:|:-:|:-:|:-:|:-:|
| D0      | 0 | 5 | 8 | 2 | 3 |...|
| C1      | 5 | 0 | 7 | 5 | 4 |...|
| C2      | 8 | 7 | 0 | 10 | 6 |...|
| M1      | 2 | 5 | 10 | 0 | 6 |...|
| M2      | 3 | 4 | 6 | 6 | 0 |...|
| ...     |...|...|...|...|...|...|

## Setup & Usage

Follow these steps to run the application locally.

### Prerequisites

-   Python 3.7+
    
-   `pip` package manager
    

### 1. Clone the Repository

```
git clone https://github.com/Takiyou14/VRP-SPD-Genetic-Algorithm.git
cd VRP-SPD-Genetic-Algorithm

```

### 2. Install Dependencies

Install the required Python libraries.

```
pip install streamlit pandas numpy

```

### 3. Run the Streamlit App

In your terminal, run the following command:

```
streamlit run app.py

```

Your web browser will automatically open a new tab with the running application.

### 4. Use the Application

1.  Upload your `demands.csv`, `stocks.csv`, and `distances.csv` files using the file uploaders in the sidebar.
    
2.  Set the desired **Vehicle Capacity**, **Population Size**, and **Number of Generations**.
    
3.  Click the **"Run Genetic Algorithm"** button.
    
4.  Observe the real-time progress and view the best solution found.
    

## Technology Stack

-   **Python:** Core programming language.
    
-   **Streamlit:** For building the interactive web application UI.
    
-   **Pandas:** For data manipulation and handling CSV files.
    
-   **Numpy:** For numerical operations.