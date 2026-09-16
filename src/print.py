import numpy as np

data = np.load("data/keypoints_normalized/fall/fall-01-cam0-rgb.npz")

print(data.files)

print(data["keypoints"])
# print(data["valid_mask"])
# print(data["frame_indices"])
# print(data["timestamps"])
# print(data["fps"])
# print(data["label"])
# print(data["video_id"])