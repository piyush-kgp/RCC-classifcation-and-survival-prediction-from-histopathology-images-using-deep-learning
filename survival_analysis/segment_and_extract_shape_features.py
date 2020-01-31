

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label,regionprops, perimeter, regionprops_table
import pandas as pd

FILE = "SAMPLE_IMAGEFOLDER/TCGA-B0-4815-01A-01-TS1.24541590-fdf9-4f7d-b6dd-9f4a06d97780/TCGA-B0-4815-01A-01-TS1.24541590-fdf9-4f7d-b6dd-9f4a06d97780_X_10240_Y_1024.png"
img_orig = cv2.imread(FILE) #0 is for b/w
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours,hierarchy = cv2.findContours(thresh, 1, 2)

label_image = np.zeros(img.shape, dtype=np.uint8)
for i, item in enumerate(contours):
    item = item.reshape(item.shape[0],2)
    for coord in item:
        x = coord[0]
        y = coord[1]
        label_image[x] = i
        label_image[y] = i

tab  = regionprops_table(label_image)
pd.DataFrame(tab).to_csv("shape_params.csv")

props = regionprops(label_image)
#'area', 'bbox', 'bbox_area', 'centroid', 'convex_area', 'convex_image', 'coords', 'eccentricity', 'equivalent_diameter', 'euler_number', 'extent', 'filled_area', 'filled_image', 'image', 'inertia_tensor', 'inertia_tensor_eigvals', 'intensity_image', 'label', 'local_centroid', 'major_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity', 'minor_axis_length', 'moments', 'moments_central', 'moments_hu', 'moments_normalized', 'orientation', 'perimeter', 'slice', 'solidity', 'weighted_centroid', 'weighted_local_centroid', 'weighted_moments', 'weighted_moments_central', 'weighted_moments_hu', 'weighted_moments_normalized'


img_cnt = cv2.drawContours(img_orig, contours, -1, (255,255,255), 3)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1),plt.title('Original Image'),plt.imshow(img_orig)#,'red')
plt.subplot(1,2,2),plt.title('OpenCV.findContours'),plt.imshow(img_cnt)#,'red')
plt.savefig("segmented.jpg")
print('number of detected contours: ',len(contours))


def main():
    # get high probability cancer patches and extract shape features from them

if __name__=="__main__":
    main()
