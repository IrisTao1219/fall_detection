import numpy as np

data = np.load("data/keypoints/fall/fall-01-cam0-rgb.npz")

print(data.files)

keypoints = data["keypoints"]
print("shape: " + str(keypoints.shape))
print("dimensions: " + str(keypoints.ndim))
print(keypoints)
# print(data["valid_mask"])
# print(data["frame_indices"])
# print(data["timestamps"])
# print(data["fps"])
# print(data["label"])
# print(data["video_id"])