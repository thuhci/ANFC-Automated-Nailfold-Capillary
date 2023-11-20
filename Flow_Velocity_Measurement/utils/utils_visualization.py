import os

import matplotlib.pyplot as plt
import seaborn as sns


def draw_scatter(sinogram, pts, output_dir, name="scatter.png"):
    if len(sinogram) == 0:
        return 
    # Extract x and y coordinates from pts
    x = [pt[1] for pt in pts]
    y = [pt[0] for pt in pts]

    # Set seaborn style for a more visually appealing plot
    sns.set(style="whitegrid", font_scale=1.2)

    # Create a heatmap with seaborn
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    sns.heatmap(sinogram, cmap="coolwarm")  # Remove color bar for simplicity

    # Add scatter plot on top of the heatmap
    area = 15 ** 2
    plt.scatter(x, y, s=area, alpha=0.1, c='black')  # Adjust color and transparency as needed

    # Set axis labels
    # plt.title("Sinogram")
    plt.xlabel("Curvilinear Distance (pixel)")
    plt.ylabel("Frame")
    
    # Set ticks on both axes
    plt.xticks([0, sinogram.shape[1] / 4, sinogram.shape[1] / 2, 3 * sinogram.shape[1] / 4, sinogram.shape[1]], ["0", "pi/4", "pi/2", "3pi/4", "pi"])
    plt.yticks([i for i in range(0, sinogram.shape[0], 10)], [str(i) for i in range(0, sinogram.shape[0], 10)])

    # Save the figure
    plt.savefig(os.path.join(output_dir, name), bbox_inches='tight')  # Use bbox_inches to prevent clipping labels
    # print(f"Saved figure at {os.path.join(output_dir, name)}")
    plt.close()
    
    
def draw_profiles(profiles, output_dir, name="profiles.png"):
    if len(profiles) == 0:
        return 
    # Set seaborn style for a more visually appealing plot
    sns.set(style="whitegrid", font_scale=1.2)

    # Create a heatmap with seaborn
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    sns.heatmap(profiles, cmap="viridis", cbar_kws={'label': 'Intensity'}, xticklabels=20, yticklabels=10)

    # Set title, axis labels
    # plt.title("Spatio-Temporal Profile")
    plt.xlabel("Curvilinear Distance (pixel)")
    plt.ylabel("Frame")

    # Save the figure
    plt.savefig(os.path.join(output_dir, name), bbox_inches='tight')  # Use bbox_inches to prevent clipping labels
    # print(f"Saved figure at {os.path.join(output_dir, name)}")
    plt.close()


def draw_boxplot(velocity_ums, output_dir, name="scatter_with_boxplot.png"):
    if len(velocity_ums) == 0:
        return 
    # Set seaborn style for a more visually appealing plot
    sns.set(style="whitegrid", font_scale=1.2)

    # Create a scatter plot with seaborn
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

    # Create a boxplot on the same axes
    sns.boxplot(x=velocity_ums, color="skyblue", showfliers=False, width=0.2, dodge=True, zorder=1)

    # Create a scatter plot after the boxplot to ensure visibility
    sns.scatterplot(x=velocity_ums, y=[1] * len(velocity_ums), color="black", marker="o", s=50, label="Data", zorder=2)

    # Annotate the normal range
    plt.axvline(x=450, color="red", linestyle="--", label="Normal Range (Lower)")
    plt.axvline(x=1200, color="green", linestyle="--", label="Normal Range (Upper)")

    # Set title, axis labels, and legend
    plt.title("Velocity Scatter Plot")
    plt.xlabel("Velocity (um/s)")
    # plt.ylabel("Boxplot as Background")
    plt.yticks([])  # Hide y-axis ticks
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(output_dir, name))
    print(f"Saved figure at {os.path.join(output_dir, name)}")
    plt.close()
    
    
def draw_boxplot_and_annotate(velocity_ums, output_dir, name="boxplot.png"):
    if len(velocity_ums) == 0:
        return 
    # Set seaborn style for a more visually appealing plot
    sns.set(style="whitegrid", font_scale=1.2)

    # Create a boxplot with seaborn
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    ax = sns.boxplot(y=velocity_ums, color="skyblue")

    # Annotate the normal range
    ax.axhline(y=450, color="red", linestyle="--", label="Normal Range (Lower)")
    ax.axhline(y=1200, color="green", linestyle="--", label="Normal Range (Upper)")
    
    # Annotate each data point with a circle
    for velocity_um in velocity_ums:
        plt.scatter(1, velocity_um, color="black", s=50, zorder=5)
        
    # Set title, axis labels, and legend
    plt.title("Velocity Boxplot")
    plt.xlabel("X-axis label")
    plt.ylabel("Velocity (um/s)")
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(output_dir, name))
    # print(f"Saved figure at {os.path.join(output_dir, name)}")
    plt.close()


def draw_wbc(wbc_time, velocity_ums, frame_num, fps, output_dir, name="WBC_events.png"):
    if len(wbc_time) > 0 and len(wbc_time) == len(velocity_ums):
        plt.title("WBC_events")
        plt.xlabel("Frame")
        plt.ylabel("WBC velocity(um/s)")
        plt.xlim(0, frame_num)
        plt.ylim(min(450, -100+min(velocity_ums)), max(1200, 100+max(velocity_ums)))
        
        plt.scatter(wbc_time, velocity_ums, marker="*")

        plt.axhline(450,  alpha=0.7, linestyle='dashdot')
        plt.axhline(1200, alpha=0.7, linestyle='dashdot')

        x = [range(frame_num)]
        for t,v in zip(wbc_time, velocity_ums):
            plt.plot(x, v*(x-t)+v, alpha=0.4, linestyle='dashed')

        plt.savefig(os.path.join(output_dir, name))
        # print(f"Saved figure at {os.path.join(output_dir, name)}")
        plt.close()

