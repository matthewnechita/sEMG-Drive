import numpy as np
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

fe = FeatureExtractor()

# load filtered data
emg_data = np.loadtxt("example_right_now.csv", delimiter=",")

# Split filtered emg into windows
windows = get_windows(emg_data, 200, 100) # 200 samples, 100 step, 50% overlap

# Extract a list of features
feature_list = ['MAV', 'SSC', 'ZC', 'WL'] # list we choose
features_1 = fe.extract_features(feature_list, windows)

# Extract a predefined feature group
features_2 = fe.extract_feature_group('HTD', windows) # feature group defined by libEMG

