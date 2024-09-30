import cv2
import numpy as np

def tune_hsv():
    """Helper function to tune the HSV values for cone detection."""
    def nothing(x):
        pass

    # Load the image for tuning
    image = cv2.imread('original.png')
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a window
    cv2.namedWindow('Trackbars')

    # Create trackbars for HSV thresholds
    cv2.createTrackbar('H Lower', 'Trackbars', 0, 179, nothing)
    cv2.createTrackbar('H Upper', 'Trackbars', 179, 179, nothing)
    cv2.createTrackbar('S Lower', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('S Upper', 'Trackbars', 255, 255, nothing)
    cv2.createTrackbar('V Lower', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('V Upper', 'Trackbars', 255, 255, nothing)

    while True:
        # Get current positions of the trackbars
        h_lower = cv2.getTrackbarPos('H Lower', 'Trackbars')
        h_upper = cv2.getTrackbarPos('H Upper', 'Trackbars')
        s_lower = cv2.getTrackbarPos('S Lower', 'Trackbars')
        s_upper = cv2.getTrackbarPos('S Upper', 'Trackbars')
        v_lower = cv2.getTrackbarPos('V Lower', 'Trackbars')
        v_upper = cv2.getTrackbarPos('V Upper', 'Trackbars')

        # Define lower and upper bounds for HSV mask
        lower_bound = np.array([h_lower, s_lower, v_lower])
        upper_bound = np.array([h_upper, s_upper, v_upper])

        # Threshold the HSV image to get only the desired color
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Show the mask
        cv2.imshow('Mask', mask)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Return the tuned HSV range
    return lower_bound, upper_bound

# Step 1: Preprocess the image
image = cv2.imread('original.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# DEBUG: Tune HSV range
#lower_orange, upper_orange = tune_hsv()

lower_orange = np.array([0, 187, 118]) 
upper_orange = np.array([179, 255, 255])  

# Get only the cones
mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

cv2.imwrite('mask.png', mask)

# DEBUG: Mask display
#cv2.imshow('Mask', mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Step 2: Detect edges
edges = cv2.Canny(mask, 50, 100)

# Step 3: Find contours of the cones
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#DEBUG
#contour_debug_image = image.copy()
#cv2.drawContours(contour_debug_image, contours, -1, (0, 255, 0), 2)  # Draw contours in green
#cv2.imshow('Contours', contour_debug_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# DEBUG: Number of contours found
#print(f"Number of contours found: {len(contours)}")

# Filter and get the centers of the cones
cone_centers = []
for contour in contours:
    if cv2.contourArea(contour) > 35:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cone_centers.append((cX, cY))

# DEBUG: Cone centers
#print("Cone Centers: ", cone_centers)

# Step 4: Fit lines to the left and right boundaries of the path
if len(cone_centers) >= 4:
    # Split left and right cones
    cone_centers.sort(key=lambda x: x[0])

    left_cones = np.array(cone_centers[:len(cone_centers)//2])
    right_cones = np.array(cone_centers[len(cone_centers)//2:])

    # Fit a line to the left and right cones
    [vx_left, vy_left, x_left, y_left] = cv2.fitLine(left_cones, cv2.DIST_L2, 0, 0.01, 0.01)
    [vx_right, vy_right, x_right, y_right] = cv2.fitLine(right_cones, cv2.DIST_L2, 0, 0.01, 0.01)

    # Step 5: Draw the lines on the image
    def draw_line(img, vx, vy, x, y, color):
        lefty = int((-x * vy / vx) + y)
        righty = int(((img.shape[1] - x) * vy / vx) + y)
        cv2.line(img, (img.shape[1] - 1, righty), (0, lefty), color, 3)

    # Draw the left and right lines
    draw_line(image, vx_left, vy_left, x_left, y_left, (0, 0, 255))
    draw_line(image, vx_right, vy_right, x_right, y_right, (0, 0, 255))

# Step 6: Save and visualize the result
cv2.imwrite('answer.png', image)
cv2.imshow('Detected Path', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
