# this is a file that accumulates the functions created in each section
#In this way, we can import then in the subsequent sections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler




def remove_correlated_features(X_train, X_test, correlation_threshold=0.8):

    # Compute correlation matrix from training data
    corr = X_train.corr().abs()

    # Get upper triangle mask (pairwise correlations only once)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Find columns to drop
    to_be_deleted = [
        column for column in upper.columns
        if any(upper[column] > correlation_threshold)
    ]

    print(len(to_be_deleted), "features removed")

    # Remove from train and test
    X_train = X_train.drop(columns=to_be_deleted)
    X_test = X_test.drop(columns=to_be_deleted)

    return X_train, X_test

def get_distinct(x):
    return(len(set(x)))

def df_scaling(df_trainC):
    # define the scaler
    scaler = StandardScaler()

    # for each column in the dataset, fit and transform the data
    for col in df_trainC.columns:
        
        # fit the scaler on the data 
        scaler.fit(df_trainC[col].values.reshape(-1, 1))

        # transform the data
        df_trainC[col] = scaler.transform(df_trainC[col].values.reshape(-1, 1))
        

def plot_corrMat(correlation_matrix, corr_lim=None):
    #if corr_lim passed, plot only correlations above limit
    
    if corr_lim:  
        filtered = correlation_matrix[correlation_matrix>corr_lim]
        columns_filtered = list(filtered.index)

        for i in filtered:
            if (filtered[i].values[np.logical_not(np.isnan(filtered[i].values))] == [1]).all():
                columns_filtered.remove(i)

        _ = len(columns_filtered)

        # Compute the heatmap
        plt.figure(figsize=(20,20))
        sns.heatmap(correlation_matrix.loc[columns_filtered, columns_filtered], cmap='Blues', vmin=.0, vmax=1, cbar_kws={'label':'Correlation'})
        plt.xlabel('Feature')
        plt.ylabel('Feature')
        plt.title(f'{_} highly correlated')
        plt.show()
        
        
    else: 
        plt.figure(figsize=(24,24))
        sns.heatmap(correlation_matrix, cmap='Blues', vmin=0.8, vmax=1, cbar_kws={'label':'Correlation'})
        plt.tight_layout()
        plt.show()
    
def print_describe_side_by_side(*dataframes, names=None, title=None):
    """
    Imprime estatísticas describe() lado a lado
    
    Parameters:
    *dataframes: DataFrames a serem comparados
    names: lista de nomes para cada DataFrame
    """
    
    stats = [df.describe() for df in dataframes]

    stats_prefixed = []
    
    print( type(stats[1]))
    combined = pd.DataFrame({name:stat for name, stat in zip(names, stats)})
    
    

    # Formata a saída
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    
    print("\n" + "="*80)
    print("Statistics - ", title if title else "")
    print("="*80)
    print(combined.to_string())
    print("="*80)
    
    return

def random_undersample(df, target_col):
    """
    Randomly undersample majority class to match minority class count
    """
    # Count samples per class
    class_counts = df[target_col].value_counts()
    minority_class = class_counts.idxmin()
    minority_count = class_counts.min()
    
    print(f"Minority class: '{minority_class}' with {minority_count} samples")
    
    # Create balanced dataset
    balanced_dfs = []
    
    for class_label in df[target_col].unique():
        class_df = df[df[target_col] == class_label]
        
        if class_label == minority_class:
            # Keep all minority samples
            balanced_dfs.append(class_df)
        else:
            # Randomly sample majority class to match minority count
            sampled_df = class_df.sample(n=minority_count, random_state=42, replace=False)
            balanced_dfs.append(sampled_df)
    
    # Combine and shuffle
    df_balanced = pd.concat(balanced_dfs, axis=0)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced

def random_undersample(df, target_col, ratio=1.0, minority_class=None, random_state=42):
    """
    Undersample all non-minority classes to achieve desired ratio relative to minority class.
    Handles multi-class scenarios.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    ratio : float, optional (default=1.0)
        Desired ratio of all other classes:minority samples
    minority_class : optional
        Specify which class is minority if auto-detection fails
    random_state : int, optional (default=42)
        Random seed for reproducibility
        
    Returns:
    --------
    pandas DataFrame with undersampled classes
    """
    # Count samples per class
    class_counts = df[target_col].value_counts()
    
    # Identify minority class
    if minority_class is None:
        minority_class = class_counts.idxmin()
    minority_count = class_counts[minority_class]

    # Calculate target counts for each class
    target_counts = {}
    for class_label, count in class_counts.items():
        if class_label == minority_class:
            target_counts[class_label] = count  # Keep all minority samples
        else:
            target_counts[class_label] = min(count, int(minority_count * ratio))
    
    print(f"\nTarget ratio for non-minority classes: {ratio}:1")
    print("Target counts:")
    for class_label, target_count in target_counts.items():
        print(f"  Class '{class_label}': {target_count} samples")
    
    # Create balanced dataset
    balanced_dfs = []
    
    for class_label in df[target_col].unique():
        class_df = df[df[target_col] == class_label]
        target_count = target_counts[class_label]
        
        if target_count == len(class_df):
            # Keep all samples
            balanced_dfs.append(class_df)
        else:
            # Undersample to target count
            sampled_df = class_df.sample(n=target_count, 
                                         random_state=random_state, 
                                         replace=False)
            balanced_dfs.append(sampled_df)
    
    # Combine and shuffle
    df_balanced = pd.concat(balanced_dfs, axis=0)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    
    return df_balanced

