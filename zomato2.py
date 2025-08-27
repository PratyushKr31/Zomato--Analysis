import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os

# Set Seaborn style
sns.set_style("whitegrid")
sns.set_palette("muted")

def load_data(file_path="Zomato-data-.csv"):
    """Load CSV into DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def clean_data(df):
    """Clean important columns in dataset."""
    if df is None:
        return None

    # Handle ratings like '3.8/5'
    def clean_rate(x):
        try:
            return float(str(x).split('/')[0])
        except:
            return np.nan
    df['rate'] = df['rate'].apply(clean_rate)

    # Convert columns
    df['online_order'] = df['online_order'].astype('category')
    df['approx_cost(for two people)'] = pd.to_numeric(
        df['approx_cost(for two people)'].astype(str).str.replace(',', ''), errors='coerce'
    )

    return df

def max_votes_restaurant(df):
    """Find restaurant with maximum votes."""
    top = df.loc[df['votes'].idxmax(), ['name', 'votes', 'listed_in(type)']]
    print("\nRestaurant with Maximum Votes:\n", top)

def rating_ttest(df):
    """Compare ratings: Online Order vs No."""
    yes = df[df['online_order'] == 'Yes']['rate'].dropna()
    no = df[df['online_order'] == 'No']['rate'].dropna()
    stat, p = ttest_ind(yes, no, equal_var=False)
    print(f"\nT-test Result: stat={stat:.2f}, p={p:.4f}")
    print("Significant difference!" if p < 0.05 else "No significant difference.")

def plot_restaurant_type(df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8,5))
    sns.countplot(x='listed_in(type)', data=df, order=df['listed_in(type)'].value_counts().index)
    plt.title("Restaurant Type Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "restaurant_types.png"))
    plt.show()

def plot_rating_distribution(df, save_dir="plots"):
    plt.figure(figsize=(8,5))
    plt.hist(df['rate'].dropna(), bins=5, edgecolor='black')
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rating_histogram.png"))
    plt.show()

def main():
    df = load_data()
    df = clean_data(df)
    if df is not None:
        max_votes_restaurant(df)
        rating_ttest(df)
        plot_restaurant_type(df)
        plot_rating_distribution(df)

if __name__ == "__main__":
    main()
