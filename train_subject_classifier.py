import numpy as np
import pandas as pd


# Inputs
cheaters = [780,1045,1551,1548,1549,1550,1173]
bad_players = [1164,664,933,219,961,596,1378,501]
good_players = [1131,1176,350,279,758,969,431, 339,1420,331,1346,878,540,607,1221,686,849, 984,355,931,790,575,1425,1099,347, 743,522,293,264, 976, 988, 619, 869, 1417, 1294, 707, 329, 930, 952, 270, 1382, 1441, 1391, 1486, 404, 1430, 317, 855, 703, 945, 708, 1354, 525, 1124, 182, 783, 222, 870, 326, 382, 434, 701, 1339, 367, 611, 1063, 1042, 385, 694, 625, 1006, 370, 463, 1258, 852, 1278,1002, 671,1076, 1016,729,337,420,1061,281, 368,811,1485,566]
catch_thresh = 0.95
data_file = "clickme_datasets/prj_clickmev2_train_imagenet_10_10_2024.npz"

# Prepare indices
cheaters_and_bad_players = np.concatenate([cheaters, bad_players])
good_players = np.asarray(good_players)
n = len(cheaters_and_bad_players)
np.random.seed(42)
trimmed_good_players = good_players[np.random.permutation(len(good_players))[:n]]

# Get data
data = np.load(data_file, allow_pickle=True)
image_path = data["file_pointer"]
clickmap_x = data["clickmap_x"]
clickmap_y = data["clickmap_y"]
user_id = data["user_id"]
user_catch_trial = data["user_catch_trial"]

# Filter subjects by catch trials
catch_trials = user_catch_trial >= catch_thresh
image_path = image_path[catch_trials]
clickmap_x = clickmap_x[catch_trials]
clickmap_y = clickmap_y[catch_trials]
user_id = user_id[catch_trials]
print("Catch trial filter from {} to {}".format(len(user_catch_trial), catch_trials.sum()))

# Combine clickmap_x/y into tuples to match Jay's format
clicks = [list(zip(x, y)) for x, y in zip(clickmap_x, clickmap_y)]

# Create dataframe
import pdb; pdb.set_trace()
df = pd.DataFrame({"image_path": image_path, "clicks": clicks, "user_id": user_id})

# Close npz
del data.f
data.close()  # avoid the "too many files are open" error

# Get a label vector
user_id = df["user_id"]
keep_ids = np.in1d(user_id, cheaters_and_bad_players) + np.in1d(user_id, trimmed_good_players)

