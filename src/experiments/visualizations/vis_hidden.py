# Leonardo Barazza, acse-lb1223

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Visualize the hidden states of the specified layer using PCA.
def visualize_hidden_states(df, layer=0, hidden_state_dim=64, color_by_codebook=True):

    # Columns for the specified hidden layer
    hidden_cols = [f'hidden-{layer}_{i}' for i in range(hidden_state_dim)]
    codebook_index_col = f'hidden-{layer}_codebook-id'

    # Extract hidden states and codebook indices for the specified hidden layer
    hidden_states = df[hidden_cols].values
    codebook_indices = df[codebook_index_col].values

    # Perform PCA
    pca = PCA(n_components=2)
    hidden_states_2d = pca.fit_transform(hidden_states)

    # Create a DataFrame for the 2D hidden states
    df_2d = pd.DataFrame(hidden_states_2d, columns=['dim1', 'dim2'])
    df_2d['codebook_index'] = codebook_indices

    # Plot the results
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of the 2D hidden states by coloring them if specified
    if color_by_codebook:
        for codebook_index in np.unique(codebook_indices):
            indices = df_2d['codebook_index'] == codebook_index
            plt.scatter(df_2d.loc[indices, 'dim1'], df_2d.loc[indices, 'dim2'], label=f'Codebook Index {int(codebook_index)}', alpha=0.1)
    else:
        plt.scatter(df_2d['dim1'], df_2d['dim2'], alpha=0.1)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'PCA Visualization of Hidden States (Layer {layer}) by {"Codebook Index" if color_by_codebook else "Single Color"}')
    if color_by_codebook:
        plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    # Load the DataFrame of hidden states
    df = pd.read_csv('experiments/visualizations/data/trajectories.csv')
    visualize_hidden_states(df, layer=0, hidden_state_dim=64, color_by_codebook=True)