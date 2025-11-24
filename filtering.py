import numpy as np
from libemg import filtering

# LOAD IN THE RAW EMG DATA
def load_emg_data(file_path):
    '''
    Load the raw EMG data from the .npz file
    Format of the .npz file {"emg": array, "fs": sampling_rate}
    '''
    data = np.load(file_path)
    emg = data["emg"]
    fs = data["fs"]

    return emg, fs

# DEFINE THE FILTERS USING libEMG
def define_filters(fs):
    '''
    Create the filter object and install:
    - Notch filter @ 60 Hz
    - Bandpass filter @ 20-450 Hz
    '''

    fi = filtering.Filter(fs)

    notch = {"name": "notch", "cutoff": 60, "bandwidth": 3}
    bandpass = {"name": "bandpass", "cutoff": [20, 450], "order": 4}

    fi.install_filter(notch)
    fi.install_filter(bandpass)

    return fi

# APPLY THE FILTERS TO THE EMG DATA
def apply_filters(fi, emg):
    '''
    This method applies the defined filters onto the raw EMG data
    '''
    filtered_data = fi.filter(emg)
    
    return np.array(filtered_data)

# SAVE THE FILTERED DATA INTO A .npz FILE
def save_filtered_data(output_path, filtered, fs):
    '''
    This method saves the filtered data into a .npz file for later use
    '''
    np.savez(output_path, emg = filtered, fs = fs)

if __name__ == "__main__":
    input_file = "emg_subjectS01_session-1.npz"
    output_file = "emg_subjectS01_session-1_filtered.npz"

    emg, fs = load_emg_data(input_file)

    fi = define_filters(fs)

    filtered_emg = apply_filters(fi, emg)

    save_filtered_data(output_file, filtered_emg, fs)
    