# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_dataset():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Explore the structure of the dataset
    print("\nDataset information:")
    print(df.info())

    # Check for missing values
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

    # Clean the dataset (no missing values in Iris dataset, but this is a placeholder)
    df_cleaned = df.dropna()  # Drop rows with missing values (if any)
    return df_cleaned

# Task 2: Basic Data Analysis
def perform_data_analysis(df):
    # Compute basic statistics
    print("\nBasic statistics of numerical columns:")
    print(df.describe())

    # Group by species and compute mean of numerical columns
    print("\nMean of numerical columns grouped by species:")
    print(df.groupby('species').mean())

    # Identify patterns or findings
    print("\nObservations:")
    print("- Setosa has the smallest petal and sepal dimensions.")
    print("- Virginica has the largest petal and sepal dimensions.")
    print("- Versicolor is intermediate in size between setosa and virginica.")

# Task 3: Data Visualization
def visualize_data(df):
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Line chart (example: sepal length over index)
    plt.figure(figsize=(10, 5))
    plt.plot(df['sepal length (cm)'], label='Sepal Length')
    plt.title('Sepal Length Over Index')
    plt.xlabel('Index')
    plt.ylabel('Sepal Length (cm)')
    plt.legend()
    plt.show()

    # Bar chart (average sepal length per species)
    plt.figure(figsize=(8, 5))
    df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Average Sepal Length (cm)')
    plt.show()

    # Histogram (distribution of petal length)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['petal length (cm)'], bins=15, kde=True, color='purple')
    plt.title('Distribution of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.show()

    # Scatter plot (sepal length vs. petal length)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='viridis')
    plt.title('Sepal Length vs. Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.show()

# Main function to execute tasks
def main():
    try:
        # Task 1: Load and explore the dataset
        df = load_and_explore_dataset()

        # Task 2: Perform basic data analysis
        perform_data_analysis(df)

        # Task 3: Visualize the data
        visualize_data(df)

    except FileNotFoundError:
        print("Error: The dataset file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the program
if __name__ == "__main__":
    main()