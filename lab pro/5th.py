import numpy as np

usd = np.array([10, 20, 50, 100, 200])

inr = usd * 83.5
discounted = inr * 0.9

print("USD Prices:", usd)
print("Prices in INR:", inr)
print("Discounted Prices:", discounted)

# Verification
check = discounted == 0.9 * 83.5 * usd
print("Check:", check)
