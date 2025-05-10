import pandas as pd
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data model file
data_file_path = os.path.join("pizza_data", "Data Model - Pizza Sales.xlsx")
pizza_data = pd.read_excel(data_file_path)
print("Pizza sales data loaded successfully. First 5 rows:")
print(pizza_data.head())

# Function to implement newsvendor model
def newsvendor_model(demand_data, co, cu, confidence_interval=0.95):
    """
    Implements the newsvendor model for optimal inventory decisions
    
    Parameters:
    demand_data (array-like): Historical demand data
    co (float): Cost of overage (cost per unit of excess inventory)
    cu (float): Cost of underage (cost per unit of unmet demand)
    confidence_interval (float): Confidence interval for plotting
    
    Returns:
    dict: Dictionary containing optimal order quantity and related statistics
    """
    # Calculate mean and standard deviation of demand
    mean_demand = np.mean(demand_data)
    std_demand = np.std(demand_data)
    
    # Calculate critical ratio
    critical_ratio = cu / (co + cu)
    
    # Calculate optimal order quantity using the normal distribution
    optimal_q = norm.ppf(critical_ratio, mean_demand, std_demand)
    
    # Calculate expected profit
    expected_profit = -co * (optimal_q - mean_demand) * norm.cdf((optimal_q - mean_demand)/std_demand) - \
                      cu * (mean_demand - optimal_q) * (1 - norm.cdf((optimal_q - mean_demand)/std_demand))
    
    # Calculate service level
    service_level = norm.cdf((optimal_q - mean_demand)/std_demand)
    
    # Prepare results
    results = {
        'mean_demand': mean_demand,
        'std_demand': std_demand,
        'optimal_order_quantity': optimal_q,
        'critical_ratio': critical_ratio,
        'service_level': service_level,
        'expected_profit': expected_profit
    }
    
    # Plot the demand distribution and optimal quantity
    x = np.linspace(mean_demand - 3*std_demand, mean_demand + 3*std_demand, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, norm.pdf(x, mean_demand, std_demand), 'b-', lw=2, label='Demand Distribution')
    plt.axvline(x=optimal_q, color='r', linestyle='--', label=f'Optimal Order Quantity: {optimal_q:.2f}')
    plt.axvline(x=mean_demand, color='g', linestyle='-', label=f'Mean Demand: {mean_demand:.2f}')
    
    # Shade the area under the curve for underage and overage
    plt.fill_between(x[x <= optimal_q], 0, norm.pdf(x[x <= optimal_q], mean_demand, std_demand), 
                    color='green', alpha=0.3, label=f'In Stock Probability: {service_level:.2%}')
    plt.fill_between(x[x > optimal_q], 0, norm.pdf(x[x > optimal_q], mean_demand, std_demand), 
                    color='red', alpha=0.3, label=f'Stockout Probability: {1-service_level:.2%}')
    
    plt.title('Newsvendor Model - Demand Distribution and Optimal Order Quantity')
    plt.xlabel('Demand/Order Quantity')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    
    return results

# Prepare data for newsvendor model
print("\nPreparing pizza sales data for newsvendor model...")

# Group data by pizza_name to analyze demand by pizza type
if 'pizza_name' in pizza_data.columns and 'quantity' in pizza_data.columns:
    # Group by pizza name and date if available
    if 'order_date' in pizza_data.columns:
        # Convert order_date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(pizza_data['order_date']):
            pizza_data['order_date'] = pd.to_datetime(pizza_data['order_date'])
            
        # Group by date and pizza_name to get daily demand
        daily_demand = pizza_data.groupby(['order_date', 'pizza_name'])['quantity'].sum().reset_index()
        
        # Get a list of unique pizza names
        pizza_names = pizza_data['pizza_name'].unique()
        
        print(f"\nFound {len(pizza_names)} different pizza types. Applying newsvendor model to each...")
        
        # Choose top 5 pizza types by sales volume for the newsvendor model
        top_pizzas = pizza_data.groupby('pizza_name')['quantity'].sum().nlargest(5).index.tolist()
        
        # Setting costs (these would typically come from business data)
        # Assuming cost of overage (waste) is 70% of unit price and cost of underage (lost sales) is 30% of unit price
        # You should adjust these based on your actual business costs
        
        # Get average unit price by pizza type if available
        if 'unit_price' in pizza_data.columns:
            avg_unit_price = pizza_data.groupby('pizza_name')['unit_price'].mean()
        else:
            # Default values if unit price isn't available
            avg_unit_price = pd.Series({pizza: 15.0 for pizza in top_pizzas})
        
        # Create a figure for subplots
        fig, axes = plt.subplots(len(top_pizzas), 1, figsize=(12, 5*len(top_pizzas)))
        
        # Apply newsvendor model to each top pizza type
        for i, pizza_name in enumerate(top_pizzas):
            print(f"\nAnalyzing: {pizza_name}")
            
            # Get daily demand data for this pizza type
            pizza_demand = daily_demand[daily_demand['pizza_name'] == pizza_name].groupby('order_date')['quantity'].sum()
            
            # Check if we have enough data
            if len(pizza_demand) < 5:
                print(f"  Insufficient data for {pizza_name}. Skipping.")
                continue
                
            print(f"  Daily demand statistics:")
            print(f"  - Min: {pizza_demand.min()}")
            print(f"  - Max: {pizza_demand.max()}")
            print(f"  - Mean: {pizza_demand.mean():.2f}")
            print(f"  - Std Dev: {pizza_demand.std():.2f}")
            
            # Calculate cost parameters based on unit price
            unit_price = avg_unit_price.get(pizza_name, 15.0)
            cost_overage = 0.7 * unit_price  # Cost of waste/overage (70% of price)
            cost_underage = 0.3 * unit_price  # Cost of lost sales/underage (30% of price)
            
            print(f"  Cost parameters:")
            print(f"  - Unit price: ${unit_price:.2f}")
            print(f"  - Cost of overage: ${cost_overage:.2f} per unit")
            print(f"  - Cost of underage: ${cost_underage:.2f} per unit")
            
            # Apply the newsvendor model
            plt.figure(figsize=(10, 6))
            results = newsvendor_model(pizza_demand.values, cost_overage, cost_underage)
            
            # Print results
            print(f"  Newsvendor model results:")
            print(f"  - Optimal daily order quantity: {results['optimal_order_quantity']:.1f} units")
            print(f"  - Service level: {results['service_level']:.2%}")
            print(f"  - Critical ratio: {results['critical_ratio']:.2f}")
            
            plt.title(f"Newsvendor Model for {pizza_name}")
            # plt.savefig(f"newsvendor_model_{pizza_name.replace(' ', '_')}.png")
            
        print("\nNewsvendor analysis completed. See the generated plots for visualizations.")
        
    else:
        print("Order date information not found in the dataset. Need order dates to analyze daily demand patterns.")
else:
    print("Required columns 'pizza_name' and 'quantity' not found in the dataset.")

# Save overall demand distribution
plt.figure(figsize=(12, 6))
if 'quantity' in pizza_data.columns:
    plt.hist(pizza_data['quantity'], bins=30, alpha=0.7, color='blue')
    plt.title('Overall Distribution of Pizza Order Quantities')
    plt.xlabel('Order Quantity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    # plt.savefig("pizza_order_quantity_distribution.png")
    print("\nSaved distribution of order quantities to 'pizza_order_quantity_distribution.png'")

print("\nNewsvendor analysis completed.")


