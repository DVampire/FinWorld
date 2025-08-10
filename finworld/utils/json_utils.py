import json
import numpy as np

def load_json(file_path):
    with open(file_path, mode='r', encoding='utf8') as fp:
        json_dict = json.load(fp)
        return json_dict
    
def save_json(json_dict, file_path, indent=4):
    with open(file_path, mode='w', encoding='utf8') as fp:
        try:
            if indent == -1:
                json.dump(json_dict, fp, ensure_ascii=False)
            else:
                json.dump(json_dict, fp, ensure_ascii=False, indent=indent)
        except Exception as e:
            if indent == -1:
                json.dump(json_dict, fp, ensure_ascii=False)
            else:
                json.dump(json_dict, fp, ensure_ascii=False, indent=indent)

def convert_to_json_serializable(data):
    """
    Recursively converts int64 and float64 to int and float in a dictionary.

    Parameters:
    - data (dict): Input dictionary with potentially non-serializable types.

    Returns:
    - dict: Output dictionary with serializable types.
    """
    for key, value in data.items():
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            data[key] = int(value)
        elif isinstance(value, np.float64):
            data[key] = float(value)
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            data[key] = convert_to_json_serializable(value)
    return data