import pandas as pd
import numpy as np


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src as decomposition_umap


# Replace 'your_file.csv' with the path to your local CSV file
file_path = "./daw.csv"


def string_to_float_array(string_array):
    """
    Convert a string array (list of lists or NumPy array) to a float array after removing commas.
    
    Args:
        string_array: Input array (list of lists or NumPy array) containing strings.
        
    Returns:
        NumPy array with float values.
        
    Raises:
        ValueError: If any value cannot be converted to float after removing commas.
    """
    try:
        # Convert to NumPy array and remove commas from strings
        cleaned_array = np.vectorize(lambda x: x.replace(',', '') if isinstance(x, str) else x)(string_array)
        # Convert to float
        float_array = np.array(cleaned_array, dtype=float)
        return float_array
    except ValueError as e:
        print(f"Error: Could not convert to float. Ensure all values are numeric after removing commas. {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
try:
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Display the first few rows of the CSV
    print("First 5 rows of the CSV file:")
    print(df.head())
    
    # Optionally, display basic information about the CSV
    print("\nCSV Info:")
    print(df.info())
    
    # Optionally, display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
    

data = df['Price']
data = np.array(data, dtype='str')
data = string_to_float_array(data)
embed, decomposition, umap = decomposition_umap.decompose_and_embed(data,decomposition_max_n=8, decomposition_method='cdd')
# embed, decomposition, umap = decomposition_umap.decompose_and_embed(ndata,emd_max_imf=5, decomposition_method='msm')

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(data[::-1], label='Original Data')

for i in range(len(decomposition)):
    plt.plot(decomposition[i][::-1], label='Decomposed Component'+ str(i))
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(embed[0], embed[1], s=1, c='blue')


x = embed[0]
y = embed[1]

prefix = './results/'
np.save(prefix + 'stock_embed_x.npy', x)
np.save(prefix + 'stock_embed_y.npy', y)
np.save(prefix + 'stock_data.npy', data)

plt.show()