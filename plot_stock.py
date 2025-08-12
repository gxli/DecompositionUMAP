import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file named 'daw.csv'
df = pd.read_csv('daw.csv')

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')

# Convert 'Price' column to numeric, handling commas
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# Sort by date for proper plotting
df = df.sort_values(by='Date')

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], marker='o', linestyle='-')
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.tight_layout()
plt.savefig('stock_price_plot.png')