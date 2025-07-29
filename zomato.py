"""
Zomato Restaurant Data Analysis
==============================
This script performs exploratory data analysis (EDA) on a Zomato dataset to uncover insights about restaurant
ratings, votes, online order availability, cost, and types. It includes data cleaning, visualizations (countplots,
histograms, line plots, boxenplots, heatmaps, bar plots, scatter plots), statistical analysis (t-test for ratings),
and optional interactive Plotly plots. Plots are saved as high-resolution PNGs in a 'plots' directory.

Dataset: Zomato-data-.csv
Requirements: pandas, numpy, matplotlib, seaborn, scipy, plotly (optional)
Author: Pratyush kumar
Date: July 2025
License: MIT 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import plotly.express as px
import os
import argparse
import logging

# Configure logging for debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='zomato_analysis.log')

# Set Seaborn style for consistent, professional aesthetics
sns.set_style("whitegrid")
sns.set_palette("muted")

def parse_arguments():
    """Parse command-line arguments for file path and interactive plotting."""
    parser = argparse.ArgumentParser(description="Zomato Restaurant Data Analysis")
    parser.add_argument('--file-path', type=str, default=r"Zomato-data-.csv",
                        help="Path to the Zomato dataset CSV file")
    parser.add_argument('--interactive', action='store_true',
                        help="Generate interactive Plotly plots in addition to static plots")
    return parser.parse_args()

def load_data(file_path):
    """Load and return the Zomato dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logging.info(f"Initial DataFrame:\n{df.head().to_string()}")
        return df
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("CSV file is empty.")
        return None
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

def clean_data(df):
    """Clean the dataset by handling 'rate' column and checking for required columns."""
    if df is None:
        return None
    required_columns = ['name', 'online_order', 'rate', 'votes', 'approx_cost(for two people)', 'listed_in(type)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return None

    def handlerate(value):
        try:
            value = str(value).split('/')[0].strip()
            return float(value)
        except (ValueError, IndexError):
            return np.nan
    df['rate'] = df['rate'].apply(handlerate)
    
    # Ensure 'online_order' is categorical
    df['online_order'] = df['online_order'].astype('category')
    
    # Convert 'approx_cost(for two people)' to numeric, handling non-numeric values
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
    
    # Log missing values
    missing_counts = df.isna().sum()
    logging.info(f"Missing values:\n{missing_counts.to_string()}")
    
    logging.info(f"Cleaned DataFrame:\n{df.head().to_string()}")
    logging.info(f"Data types:\n{df.dtypes.to_string()}")
    return df

def find_max_votes_restaurant(df):
    """Find and print restaurant(s) with the maximum votes."""
    if df is None:
        return
    try:
        max_votes = df['votes'].max()
        restaurants = df.loc[df['votes'] == max_votes, ['name', 'votes', 'listed_in(type)']]
        logging.info(f"Restaurant(s) with maximum votes ({max_votes}):\n{restaurants.to_string()}")
        print(f"Restaurant(s) with maximum votes ({max_votes}):\n{restaurants}")
    except KeyError:
        logging.error("'votes', 'name', or 'listed_in(type)' column not found.")

def perform_statistical_analysis(df):
    """Perform t-test to compare ratings by online order status."""
    if df is None:
        return
    try:
        yes_ratings = df[df['online_order'] == 'Yes']['rate'].dropna()
        no_ratings = df[df['online_order'] == 'No']['rate'].dropna()
        if len(yes_ratings) > 0 and len(no_ratings) > 0:
            stat, p_value = ttest_ind(yes_ratings, no_ratings, equal_var=False)
            logging.info(f"T-test for ratings (Online vs. No Online): stat={stat:.4f}, p-value={p_value:.4f}")
            print(f"T-test for ratings (Online vs. No Online): stat={stat:.4f}, p-value={p_value:.4f}")
            if p_value < 0.05:
                logging.info("Significant difference in ratings (p<0.05).")
                print("Significant difference in ratings (p<0.05).")
            else:
                logging.info("No significant difference in ratings (p>=0.05).")
                print("No significant difference in ratings (p>=0.05).")
        else:
            logging.warning("Insufficient data for t-test.")
            print("Insufficient data for t-test.")
    except KeyError:
        logging.error("'online_order' or 'rate' column not found.")

def create_plots_directory(save_dir="plots"):
    """Create plots directory if it doesn't exist."""
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Plots directory created/confirmed at: {save_dir}")

def plot_restaurant_type_count(df, save_dir="plots"):
    """Plot count of restaurants by type."""
    if df is None:
        return
    plt.figure(figsize=(8, 6))
    sns.countplot(x='listed_in(type)', data=df, order=df['listed_in(type)'].value_counts().index)
    plt.title("Restaurant Type Distribution", fontsize=14)
    plt.xlabel("Restaurant Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "restaurant_type_count.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info("Saved restaurant type count plot.")

def plot_votes_by_restaurant_type(df, save_dir="plots"):
    """Plot total votes by restaurant type."""
    if df is None:
        return
    plt.figure(figsize=(8, 6))
    grouped_data = df.groupby('listed_in(type)')['votes'].sum().sort_values()
    plt.plot(grouped_data.index, grouped_data.values, color='green', marker='o', linestyle='-', linewidth=2)
    plt.title("Total Votes by Restaurant Type", fontsize=14)
    plt.xlabel("Restaurant Type", fontsize=12)
    plt.ylabel("Total Votes", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "votes_by_restaurant_type.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info("Saved votes by restaurant type plot.")

def plot_rating_histogram(df, save_dir="plots", interactive=False):
    """Plot histogram of restaurant ratings (static or interactive)."""
    if df is None:
        return
    if interactive:
        try:
            fig = px.histogram(df, x='rate', nbins=5, title="Rating Distribution",
                               labels={'rate': 'Rating', 'count': 'Frequency'})
            fig.update_layout(title_x=0.5, template='plotly_white', width=600, height=400)
            fig.update_traces(marker=dict(line=dict(color='black', width=1)))
            fig.update_xaxes(title_font=dict(size=12), tickfont=dict(size=10))
            fig.update_yaxes(title_font=dict(size=12), tickfont=dict(size=10))
            fig.update_layout(showlegend=False)
            fig.write_xaxes(range=[df['rate'].min() - 0.1, df['rate'].max() + 0.1])
            fig.write_yaxes(range=[0, df['rate'].dropna().value_counts().max() + 10])
            fig.write_layout(margin=dict(l=50, r=50, t=50, b=50))
            fig.write_layout(bargap=0.1)
            fig.write_layout(title_font=dict(size=14))
            fig.show()
            fig.write(os.path.join(save_dir, "rating_histogram_interactive.html"))
            logging.info("Saved interactive rating histogram.")
        except Exception as e:
            logging.error(f"Error generating interactive histogram: {e}")
    else:
        plt.figure(figsize=(8, 6))
        plt.hist(df['rate'].dropna(), bins=5, edgecolor='black')
        plt.title("Rating Distribution", fontsize=14)
        plt.xlabel("Rating", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rating_histogram.png"), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        logging.info("Saved rating histogram.")

def plot_cost_count(df, save_dir="plots"):
    """Plot count of restaurants by approximate cost."""
    if df is None:
        return
    plt.figure(figsize=(8, 6))
    sns.countplot(x='approx_cost(for two people)', data=df, order=df['approx_cost(for two people)'].value_counts().index)
    plt.title("Approximate Cost for Two People Distribution", fontsize=14)
    plt.xlabel("Cost for Two People (INR)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cost_count.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info("Saved cost count plot.")

def plot_rating_by_online_order(df, save_dir="plots"):
    """Plot boxenplot of ratings by online order status."""
    if df is None:
        return
    plt.figure(figsize=(8, 6))
    sns.boxenplot(x='online_order', y='rate', data=df)
    plt.title("Rating Distribution by Online Order", fontsize=14)
    plt.xlabel("Online Order", fontsize=12)
    plt.ylabel("Rating", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rating_by_online_order.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info("Saved rating by online order plot.")

def plot_heatmap(df, save_dir="plots"):
    """Plot heatmap of restaurant counts by type and online order."""
    if df is None:
        return
    plt.figure(figsize=(8, 6))
    pivot_table = df.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
    sns.heatmap(pivot_table, annot=True, cmap='Blues', fmt='d')
    plt.title("Restaurant Count by Type and Online Order", fontsize=14)
    plt.xlabel("Online Order", fontsize=12)
    plt.ylabel("Restaurant Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info("Saved heatmap.")

def plot_avg_rating_by_type(df, save_dir="plots"):
    """Plot average rating by restaurant type."""
    if df is None:
        return
    plt.figure(figsize=(8, 6))
    avg_ratings = df.groupby('listed_in(type)')['rate'].mean().sort_values()
    sns.barplot(x=avg_ratings.index, y=avg_ratings.values)
    plt.title("Average Rating by Restaurant Type", fontsize=14)
    plt.xlabel("Restaurant Type", fontsize=12)
    plt.ylabel("Average Rating", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_rating_by_type.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info("Saved average rating by type plot.")

def plot_votes_vs_rating(df, save_dir="plots"):
    """Plot votes vs. rating with restaurant type as hue."""
    if df is None:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='rate', y='votes', hue='listed_in(type)', size='approx_cost(for two people)', data=df)
    plt.title("Votes vs. Rating by Restaurant Type", fontsize=14)
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Votes", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "votes_vs_rating.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info("Saved votes vs. rating plot.")

def main():
    """Main function to execute the Zomato data analysis."""
    args = parse_arguments()
    create_plots_directory()
    
    df = load_data(args.file_path)
    if df is not None:
        df = clean_data(df)
        if df is not None:
            find_max_votes_restaurant(df)
            perform_statistical_analysis(df)
            plot_restaurant_type_count(df)
            plot_votes_by_restaurant_type(df)
            plot_rating_histogram(df, interactive=args.interactive)
            plot_cost_count(df)
            plot_rating_by_online_order(df)
            plot_heatmap(df)
            plot_avg_rating_by_type(df)
            plot_votes_vs_rating(df)

if __name__ == "__main__":
    main()