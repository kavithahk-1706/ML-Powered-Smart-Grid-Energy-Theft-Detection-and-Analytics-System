import pandas as pd
import numpy as np

def add_area_column(input_path, output_path, num_areas=5, seed=42):
    
    '''
    Docstring for add_area_column
    
    :param input_path: The input path of the raw dataset
    :param output_path: The output path for the processed dataset with area column
    :param num_areas: Integer of total number of areas
    :param seed: Value sent to np.random.seed() to prevent randomization
    
    '''

    print(f"Loading dataset from {input_path}...")
    df=pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")

    #seed to make sure the values aren't randomized on every run
    np.random.seed(42)
    df['area_id']=np.random.choice(range(1,num_areas+1), size=len(df))

    print(f"\nArea distribution: ")
    print(df['area_id'].value_counts().sort_index())

    df.to_csv(output_path, index=False)
    print(f"\n Saved to: {output_path}")
    print(f"New shape: {df.shape}")

    return df

if __name__=="__main__":
    add_area_column(
        input_path='../data/raw/raw_dataset.csv',
        output_path='../data/processed/dataset1.csv',
        num_areas=5,
        seed=42
    )