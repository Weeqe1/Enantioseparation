# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot(y_true, y_pred, mae, r2, rmse, TRANSFER_TARGET, fold):
    """
    Generate prediction plots and error histograms for different chromatographic columns.

    This function creates two types of visualizations:
    1. A hexbin scatter plot showing predicted vs real values with performance metrics
    2. An error histogram showing the distribution of prediction errors

    Args:
        y_true: Array of true retention time values
        y_pred: Array of predicted retention time values  
        mae: Mean Absolute Error
        r2: R-squared coefficient
        rmse: Root Mean Square Error
        TRANSFER_TARGET: Name of the chromatographic column being analyzed
        fold: Cross-validation fold number (-1 for ensemble results)
    """

    # Define color schemes for different chromatographic columns
    if TRANSFER_TARGET == 'Cyclosil_B':
        clist = ['#ffffff', '#8fbbda', '#1f77b4']
        facecolor = '#1f77b4'

    if TRANSFER_TARGET == 'Cyclodex_B':
        clist = ['#ffffff', '#ffbf87', '#ff7f0e']
        facecolor = '#ff7f0e'

    if TRANSFER_TARGET == 'HP_chiral_20β':
        clist = ['#ffffff', '#96d096', '#2ca02c']
        facecolor = '#2ca02c'

    if TRANSFER_TARGET == 'CP_Cyclodextrin_β_2,3,6_M_19':
        clist = ['#ffffff', '#eb9394', '#d62728']
        facecolor = '#d62728'

    if TRANSFER_TARGET == 'CP_Chirasil_D_Val':
        clist = ['#ffffff', '#cab3de', '#9467bd']
        facecolor = '#9467bd'

    if TRANSFER_TARGET == 'CP_Chirasil_Dex_CB':
        clist = ['#ffffff', '#c6aba5', '#8c564b']
        facecolor = '#8c564b'

    if TRANSFER_TARGET == 'CP_Chirasil_L_Val':
        clist = ['#ffffff', '#f1bbe1', '#e377c2']
        facecolor = '#e377c2'

    # Determine file suffix based on fold type
    if fold == -1:  # Ensemble results
        fold_suffix = "_ensemble"
    else:
        fold_suffix = f"_{fold + 1}"

    # Create custom colormap from the defined color list
    newcmp = LinearSegmentedColormap.from_list('chaos', clist)

    # Flatten arrays to ensure proper shape for plotting
    out_y_pred = np.reshape(y_pred, (-1,))
    out_y_test = np.reshape(y_true, (-1,))

    # Determine plot axis limits based on data range
    xmin = out_y_test.min()
    xmax = out_y_test.max()

    # Create main prediction plot
    fig = plt.figure(figsize=(10, 6))

    # Configure tick directions to point inward
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Set axis labels with bold formatting
    plt.xlabel('Real values for RT', fontsize=18, weight='bold')
    plt.ylabel('Predicted values for RT', fontsize=18, weight='bold')

    # Configure tick label sizes
    plt.yticks(size=16)
    plt.xticks(size=16)

    # Draw diagonal line representing perfect prediction (y=x)
    plt.plot([xmin, xmax], [xmin, xmax], ':', linewidth=4, color='red')

    # Add performance metrics text to the plot
    plt.text(xmin + (xmax - xmin) * 0.02, xmax - (xmax - xmin) * 0.05, f'MAE: {mae:.2f}', fontsize=20, weight='bold',
             color=facecolor)
    plt.text(xmin + (xmax - xmin) * 0.02, xmax - (xmax - xmin) * 0.1, f'RMSE: {rmse:.2f}', fontsize=20, weight='bold',
             color=facecolor)
    plt.text(xmin + (xmax - xmin) * 0.02, xmax - (xmax - xmin) * 0.15, f'R²: {r2:.2f}', fontsize=20, weight='bold',
             color=facecolor)

    # Create hexagonal binning plot to show data density
    plt.hexbin(out_y_test, out_y_pred, gridsize=20, extent=[xmin, xmax, xmin, xmax],
               cmap=newcmp)

    # Set axis limits to match data range
    plt.axis([xmin, xmax, xmin, xmax])

    # Configure tick marks on all sides of the plot
    ax = plt.gca()
    ax.tick_params(top=True, right=True)

    # Add colorbar to show frequency scale
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Frequency', size=18, weight='bold', rotation=270, labelpad=15)

    # Save the prediction plot
    plt.savefig(
        f'Output/GAT_model/{TRANSFER_TARGET}/pics/{TRANSFER_TARGET}_predictions{fold_suffix}.png',
        dpi=600
    )
    plt.close()

    # Create error histogram plot
    fig = plt.figure(figsize=(8, 3))
    plt.subplots_adjust(bottom=0.25)

    # Calculate prediction errors
    errors = y_true - y_pred

    # Create histogram of errors
    plt.hist(errors, bins=200, facecolor=facecolor, alpha=0.7)

    # Set axis labels for error plot
    plt.xlabel("Error", fontsize=18, weight='bold')
    plt.ylabel("Frequency", fontsize=18, weight='bold')

    # Configure tick label sizes
    plt.yticks(size=16)
    plt.xticks(size=16)

    # Save the error histogram
    plt.savefig(
        f'Output/GAT_model/{TRANSFER_TARGET}/pics/{TRANSFER_TARGET}_Error{fold_suffix}.png',
        dpi=600
    )