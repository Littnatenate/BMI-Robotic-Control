while True:
    try:
        weight = float(input("What's the package weight (in lb)? "))
        if weight > 0:
            break
        else:
            print("Weight must be a positive number.")
    except ValueError:
        print("Try again, please use numbers for the weight.")

def calculate_ground_shipping(weight):
    flat_charge = 20.00
    if weight <= 2:
        rate = 1.50
    elif 2 < weight <= 6:
        rate = 3.00
    elif 6 < weight <= 10:
        rate = 4.00
    else:
        rate = 4.75
    
    cost = (weight * rate) + flat_charge
    return cost

premium_ground_shipping_cost = 125.00

def calculate_drone_shipping(weight):
    flat_charge = 0.00
    if weight <= 2:
        rate = 4.50
    elif 2 < weight <= 6:
        rate = 9.00
    elif 6 < weight <= 10:
        rate = 12.00
    else:
        rate = 14.25
    
    cost = (weight * rate) + flat_charge
    return cost

ground_cost = calculate_ground_shipping(weight)
drone_cost = calculate_drone_shipping(weight)
premium_cost = premium_ground_shipping_cost

cheapest_cost = min(ground_cost, drone_cost, premium_cost)

if cheapest_cost == ground_cost:
    method = "Standard Ground Shipping"
elif cheapest_cost == premium_cost:
    method = "Premium Ground Shipping"
else:
    method = "Drone Shipping"

print("\n--- Sal's Shippers Cost Breakdown ---")
print(f"Package Weight: {weight:.2f} lb")
print(f"Standard Ground Shipping Cost: ${ground_cost:.2f}")
print(f"Premium Ground Shipping Cost: ${premium_cost:.2f}")
print(f"Drone Shipping Cost: ${drone_cost:.2f}")

print("\n--- Cheapest Option ---")
print(f"The cheapest option is {method}.")
print(f"The total cost will be ${cheapest_cost:.2f}.")