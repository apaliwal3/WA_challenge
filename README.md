# WA_challenge

Libraries:
- OpenCV
- NumPy

Methodology:

1. Started by importing libraries and the original image, as I had to filter the cones by their color I converted the image to a HSV color space for processing. Tuning the threshold for the mask to only detect the cones was the first challenge as just guessing the HSV range wasn't working. After some searching I found a method to create trackbars within cv2 that allowed me to interactively edit the thresholds until the mask showed mostly just the cones, there were some outliers such as the bright exit sign.
2. The next step was to detect the edges of the cones for this I used canny edge detection which detects based on gradients in image intensity. This edge detection worked fairly well after messing around with the paramenters for a bit. This however may have not been the best method as in later steps some cones could not get centers placed on them.
3. Used cv2's findContours to find the contours of the edges. Filtered the contours by area to find the centers of the cones. The issue with certain bright spots still remaining on the mask affected this once again as I couldnt lower the threshold for the area lower which may have detected more cones but also detected random bright spots such as the exit sign.  
4. Split left and right cones by splitting the image down the x axis and fit the different points from the cones into a line using fitLine which uses least squares.
5. Drew the line using line in the cv2 library.
6. Wrote the resulting image

Notes:

I believe the fitment to the cones wasn't perfect either due to the method of finding the contours and edges which didn't create whole boundaries around the cones or due to the other brights spots in the image that I could not remove without also removing the cones from the mask. 

![answer](https://github.com/user-attachments/assets/e162c404-d5f3-4f96-a9ea-fdff92d4c849)
