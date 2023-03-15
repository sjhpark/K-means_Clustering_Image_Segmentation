import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2

# Hyperparameters
n_clusters_list = np.arange(1,51) # number of clusters
max_iter = 20 # maximum number of iterations
threshold = 0.1 # threshold for convergence
eps = 1e-8 # epsilon for numerical stability
np.random.seed(42) # set seed for reproducibility

# define image directory
image_dir = "rectified_images/Kitti_Stereo_2015/data_scene_flow/testing/image_2/"

# list of images
image_list = []

for n_clusters in tqdm(n_clusters_list, desc="Image"):

    # sample image
    image_sample = image_dir + os.listdir(image_dir)[49]

    # read image
    image = cv2.imread(image_sample)

    # convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # shape of image
    width = image.shape[0] # width
    height = image.shape[1] # height 
    channels = image.shape[2] # channels

    # reshape image
    image = image.reshape(width*height, channels)

    ################### K-MEANS CLUSTERING ###################
    centers = np.zeros((n_clusters, channels)) # centroids
    centers_old = centers.copy() # centroids from previous iteration

    # Randomly initialize cluster centers (centroids)
    for i in range(n_clusters):
        # randomly pick a set of R G B pixels as each centroid
        # ex) centroid 1 = [R220, G220, B220]
        # ex) centroid 2 = [R81, G81, B81]
        centers[i] = image[np.random.choice(image.shape[0], replace=False)]

    # Update cluster centers until convergence
    for i in range(max_iter):

        # intialize previous centroids
        centers_old = centers.copy()

        # L2 distance between each of the pixels and each of the centroids
        # distances.shape = (width*height, n_clusters)
        distances = np.linalg.norm(image[:, None] - centers, axis=2)

        # indices of the closest centroids for each of the pixels
        labels = np.argmin(distances, axis=1) 

        # update centroids
        for j in range(n_clusters):
            # update: centroid_n = 1/N * Sigma(x_i) 
            # (where x_i is the pixel that belongs to centroid_n)
            centers[j] = np.mean(image[labels == j], axis=0)

            # Some pixels may not belong to any cluster and their mean will be NaN.
            # Thus, convert NaN to 0 and add epsilon.
            centers[j] = np.nan_to_num(centers[j]) + eps

        # print("distance between current center and previous center: ", np.linalg.norm(centers - centers_old))

        # check for convergence
        if np.linalg.norm(centers - centers_old) < threshold:
            print(f"Iteration {i}: all centroids have converged before max iteration.")
            break # break out of for-loop

    # Assign random color to each cluster
    segmented_data = np.zeros((width*height, channels)) # color-based segmented image
    new_colors = np.random.randint(0, 255, size=(n_clusters, channels)) # random colors
    for i in range(n_clusters):
        # assign each random color to the pixel belonging to each cluster
        segmented_data[labels == i] = new_colors[i]

    # Reshape segmented data to original image shape
    segmented_image = segmented_data.reshape((width, height, channels)).astype('uint8')

    ##############################################################

    # Append segmented image to image list
    image_list.append(segmented_image)


###################### CREATE a GIF ######################
frames = [] # frames
font = ImageFont.truetype("impact.ttf", 40) # font

# Add title and append each frame to the list of frames
for i, img in enumerate(image_list):
    # Open image
    image = Image.fromarray(img)

    # Add title as text overlay
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), f"Number of clusters: {n_clusters_list[i]}", font=font)

    # Append frame to list of frames
    frames.append(image)

# Greate and save a GIF
frames[0].save('segmented_image.gif', format='GIF', append_images=frames[1:], save_all=True, duration=1000, loop=10)
   
