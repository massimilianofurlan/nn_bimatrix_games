import os
import json
import toml
import pickle

# Input / Output utils 

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def read_config(config_file, key):
    with open(config_file, 'r') as f:
        config = toml.load(f)
    return config[key]

def list_dirs(base_dir):
    """List all unique directories within the specified base directory."""
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def print_metadata(metadata):
    for param, value in metadata.items():
        print(f"  {param}: {value}")

def log_metadata(file_path, metadata, intro, w='w'):
    with open(f'{file_path}', w) as f:
        f.write(f"{intro}\n")
        for param, value in metadata.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\n")

def load_metadata(base_dir, directory, metadata_file):
    metadata_path = os.path.join(base_dir, directory, metadata_file)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def preview_dataset(dataset_metadata, dataset):
    print_metadata(dataset_metadata)
    # Display a game from the dataset
    print("\nExample game:")
    print_bimatrix_game(*dataset[0])

def display_info(base_dir, metadata_file):
    """Display information by reading their metadata."""
    print(f"Available {base_dir}:")
    dirs = sorted(list_dirs(base_dir), key=lambda x: (isinstance(x, str), x))
    info = []
    i = 1
    for _, directory in enumerate(dirs, start=1):
        metadata_path = os.path.join(base_dir, directory, metadata_file)
        if os.path.exists(metadata_path):
            metadata = load_metadata(base_dir, directory, metadata_file)
            info.append((directory, metadata))
            print(f"{i}) {directory}:")
            print_metadata(metadata)
            print()
            i += 1
    return info

def select_item(info):
    """Allow user to select an item and load it."""
    selection = int(input("Select an item to load: ")) - 1
    if selection < 0 or selection >= len(info):
        raise ValueError("Invalid selection.")
    print()
    model_dir, metadata = info[selection]

    return model_dir, metadata
    
def print_bimatrix_game(A, B, fmt=">+7.3f"):
    # Print game matrices with the given format
    for row1, row2 in zip(A, B):
        formatted_row = ''.join(f"({a:{fmt}}, {b:{fmt}}) " for a, b in zip(row1, row2))
        print(formatted_row)

def display_model(model_metadata):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Model loaded")
    print(f"\nModel: ")
    print_metadata(model_metadata)

def print_and_log(message, file):
    print(message)
    print(message, file=file)

def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
