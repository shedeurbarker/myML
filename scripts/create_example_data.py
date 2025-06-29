import pandas as pd
import numpy as np

def create_example_data():
    # Read the first two rows of the CSV file
    file_path = 'results/extract/extracted_data.csv'
    data = pd.read_csv(file_path, nrows=55)
    
    # index to fetch the values - choose a random index
    index = np.random.randint(3, len(data))
    print(data['IntSRHn'].values[index])
    print(data['IntSRHp'].values[index])
    
    # Load feature names from the saved numpy file
    feature_names = np.load('scripts/feature_names.npy')
    
    # Create a DataFrame using the feature names
    example_data = pd.DataFrame(columns=feature_names)
    
    # Add the row data using the feature names
    row_data = {}
    for feature in feature_names:
        if feature in data.columns:
            row_data[feature] = data[feature].values[index]
        else:
            print(f"Warning: Feature {feature} not found in data")
            row_data[feature] = 0  # or some other default value
    
    # Add the row to the DataFrame
    example_data.loc[0] = row_data
    
    # Get the target variables (IntSRHn and IntSRHp)
    experimental_values = {
        'IntSRHn': data['IntSRHn'].values[index],  # Take first value since we're using first row
        'IntSRHp': data['IntSRHp'].values[index]   # Take first value since we're using first row
    }
    print(example_data)
    
    return example_data, experimental_values

# Call the function to get the DataFrame and experimental values
example_data, experimental_values = create_example_data() 