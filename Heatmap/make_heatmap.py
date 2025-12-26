import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
BAD_CODE_DIR = "directions/bad_code_directions"
REFUSAL_DIR = "directions/refusal_directions"
LAYERS = range(1, 29)  # Layers 1 to 28


OUTPUT_FILENAME = "similarity_heatmap.pdf"

def analyze_and_plot():
    """
    Loads concept direction vectors, calculates their cosine similarities
    across all layers, and plots the result as a heatmap with the
    origin (1,1) at the bottom-left corner and larger fonts for all elements.
    """
    print("--- Starting Similarity Analysis ---")
    
    # Check if directories exist
    if not os.path.isdir(BAD_CODE_DIR) or not os.path.isdir(REFUSAL_DIR):
        print(f"Error: Required direction folders not found.")
        print(f"Please run the extraction script first to generate '{BAD_CODE_DIR}' and '{REFUSAL_DIR}'.")
        return

    num_layers = len(list(LAYERS))
    similarity_matrix = np.zeros((num_layers, num_layers))

    print("Calculating cosine similarity matrix...")
    
    # Use tqdm for a progress bar
    for i in tqdm(LAYERS, desc="Bad Code Layers"):
        for j in tqdm(LAYERS, desc="Refusal Layers", leave=False):
            bad_code_path = os.path.join(BAD_CODE_DIR, f"bad_code_concept_direction_layer_{i}.pt")
            refusal_path = os.path.join(REFUSAL_DIR, f"refusal_concept_direction_layer_{j}.pt")

            try:
                vec_bad_code = torch.load(bad_code_path, map_location='cpu')
                vec_refusal = torch.load(refusal_path, map_location='cpu')
                similarity = torch.dot(vec_bad_code.flatten(), vec_refusal.flatten())
                similarity_matrix[i - 1, j - 1] = similarity.item()
            except FileNotFoundError as e:
                print(f"\nWarning: Could not find a file. {e}")
                print("Matrix will have zero-entries for missing data.")
                continue

    # --- Plotting the Heatmap ---
    print("\nGenerating heatmap with larger fonts...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 14)) 
    
    sns.heatmap(
        similarity_matrix,
        ax=ax,
        cmap='viridis',
        annot=False,
        xticklabels=[str(l) for l in LAYERS],
        yticklabels=[str(l) for l in LAYERS],
        # Add a label to the color bar for clarity
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    # Invert the Y-axis to place (1,1) at the bottom-left origin
    ax.invert_yaxis()
    
    # --- FONT SIZE CHANGES ARE HERE ---
    
    #ax.set_title('Cosine Similarity between "Malicious Code Generation" and "Refusal" Concept Directions', fontsize=24, pad=25)
    ax.set_xlabel('Refusal Concept Layer', fontsize=28)
    ax.set_ylabel('Bad Code Concept Layer', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    # Get the color bar and set its font sizes ---
    cbar = ax.collections[0].colorbar
    # Set the font size of the color bar's tick labels
    cbar.ax.tick_params(labelsize=18)
    # Set the font size of the color bar's label
    cbar.ax.yaxis.label.set_size(20)

    
    tick_interval = 2
    ax.set_xticks(np.arange(0, num_layers, tick_interval))
    ax.set_xticklabels(np.arange(1, num_layers + 1, tick_interval))
    ax.set_yticks(np.arange(0, num_layers, tick_interval))
    ax.set_yticklabels(np.arange(1, num_layers + 1, tick_interval))
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    # Saved as PDF with tight bounding box to prevent label clipping
    plt.savefig(OUTPUT_FILENAME, format='pdf', bbox_inches='tight')
    
    print(f"--- Analysis complete! Heatmap saved to '{OUTPUT_FILENAME}' ---")

if __name__ == "__main__":
    analyze_and_plot()