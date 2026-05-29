import matplotlib.pyplot as plt
import csv
import os

# --- Load data ---
# Replace 'scores.csv' with your actual CSV file path
# csv_path = './results/cifar100_results_128/clip_logo_attribution_ddim1000/clip_logo_attribution_sorted.csv'

# csv_path = "./results/cifar100_results_128/clip_logo_attribution/clip_logo_attribution_sorted.csv"
# csv_path = "./results/cifar100_results_128/clip_logo_attribution/clip_logo_attribution_random_exclude_ddpm4000/clip_logo_attribution_sorted.csv"

csv_path = "./results/cifar100_results_128/clip_logo_attribution/clip_logo_attribution_random_exclude_ddim1000/clip_logo_attribution_sorted.csv"

# Read CSV file
with open(csv_path, 'r') as f:
    csv_reader = csv.reader(f)
    headers = next(csv_reader)  # Get column headers
    
    # Find score columns (ending with '_Sc')
    score_cols_indices = [i for i, col in enumerate(headers) if col.endswith('_Sc')]
    score_cols = [headers[i] for i in score_cols_indices]
    
    # Read all data
    data = []
    for row in csv_reader:
        data.append([float(row[i]) for i in score_cols_indices])

# --- Plot settings ---
plt.figure(figsize=(10, 12))
plt.title('Distribution of R*_Sc per Image')  # Title of the plot

# --- Scatter each image's scores ---
for idx, row_data in enumerate(data):
    x = row_data                # all R*_Sc values for this image
    y = [idx] * len(x)          # same y-coordinate for this image
    plt.scatter(x, y, alpha=0.5) # semi-transparent dots

# --- Labels ---
plt.xlabel('Score')              # X-axis label
plt.ylabel('Image Index')        # Y-axis label

# Optional: invert y-axis so image_0000 is at the top
plt.gca().invert_yaxis()

plt.tight_layout()

# Save the image (PNG format) to the same directory as the CSV file
csv_dir = os.path.dirname(csv_path)
output_path = os.path.join(csv_dir, "score_ditribution_plot.png")
plt.savefig(output_path, dpi=300)
print(f"Image saved to {output_path}")
