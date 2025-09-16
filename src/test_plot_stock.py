import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data.npy' exists and contains your data
data = np.load('./data.npy')

grad = np.gradient(data)
lapl = np.gradient(grad) # This is the second derivative (Laplacian for 1D)

# Handle potential division by zero or negative values for lapl_k if necessary
# For example, if data can be zero, add a small epsilon or check for zero before division
# Or if lapl/data can be negative, decide how to handle it for sqrt.
# For simplicity, I'm proceeding assuming valid divisions and non-negative values for sqrt.
grad_k = np.abs(grad / data)
# Ensure lapl/data is non-negative before sqrt to avoid NaNs if it can be negative
lapl_k = np.sqrt(np.abs(lapl / data))

# Handle potential division by zero for lapl_rati if grad_k can be zero
# Adding a small epsilon to the denominator for stability
epsilon = 1e-10 # A small number to prevent division by zero
lapl_rati = lapl_k / (grad_k + epsilon)

plt.figure(figsize=(10, 6)) # Make the figure a bit larger for better visibility
plt.plot(data, label='Original Data')
plt.plot(grad_k, label='Normalized Gradient ($|grad/data|$)', linestyle='--')
plt.plot(lapl_k, label='Normalized Laplacian ($\sqrt{|lapl/data|}$)', linestyle=':')
plt.plot(lapl_rati, label='Laplacian Ratio ($lapl\_k/grad\_k$)', color='red')

# --- Highlight regions where lapl_rati > 1 ---
# Find the indices where lapl_rati is greater than 1
where_large_ratio = np.where(lapl_rati > 1)[0]

# Group consecutive indices to draw continuous shaded regions
if len(where_large_ratio) > 0:
    start_idx = where_large_ratio[0]
    for i in range(1, len(where_large_ratio)):
        if where_large_ratio[i] != where_large_ratio[i-1] + 1:
            # End of a contiguous block, draw the shaded region
            plt.axvspan(start_idx, where_large_ratio[i-1] + 1, color='green', alpha=0.2, label='Ratio > 1' if start_idx == where_large_ratio[0] else "")
            start_idx = where_large_ratio[i]
    # Draw the last contiguous block
    plt.axvspan(start_idx, where_large_ratio[-1] + 1, color='green', alpha=0.2, label='_nolegend_' if start_idx != where_large_ratio[0] else "") # Use _nolegend_ for subsequent spans

plt.title('Data, Derivatives, and Normalized Ratios')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()