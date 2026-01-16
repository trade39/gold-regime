import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_price_and_regimes(data, probs):
    """
    Plots the gold price and the smoothed probabilities of the regimes.
    
    Args:
        data (pd.DataFrame): Dataframe with 'Price'. Index should be datetime.
        probs (pd.DataFrame): Smoothed probabilities from the model.
    """
    # Use a style for better aesthetics
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot Gold Price
    axes[0].plot(data.index, data['Price'], color='#D4AF37', label='Gold Price (USD)', linewidth=1.5)
    axes[0].set_title('Gold Price and Regime Probabilities', fontsize=16)
    axes[0].set_ylabel('Price (USD)', fontsize=12)
    axes[0].legend(loc='upper left')
    
    # Plot Regime Probabilities
    cols = probs.columns
    # Define a color palette (up to 5 regimes)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, col in enumerate(cols):
        color = colors[i % len(colors)]
        axes[1].plot(probs.index, probs[col], label=f'Regime {col}', color=color, alpha=0.6, linewidth=1)
        axes[1].fill_between(probs.index, 0, probs[col], color=color, alpha=0.1)
    
    axes[1].set_ylabel('Probability', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc='upper left')
    
    plt.tight_layout()
    return fig
