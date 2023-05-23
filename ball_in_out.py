"""
import matplotlib.path as mplPath
from event_handler import click_event
import cv2

img = cv2.imread('court1.png', 1)
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
# Define the four corners of the court
corners=click_event()
print(corners)
#corners = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]  # Replace with your coordinates

# Create a polygon
poly_path = mplPath.Path(corners)

# Define the centroid of the ball
#ball_centroid = (ball_x, ball_y)  # Replace with your coordinates

# Check if the centroid is inside the court

if poly_path.contains_point(ball_centroid):
    print("The ball is in.")
else:
    print("The ball is out.")
    
    """
"""
import cv2
from event_handler import click_on_image

# Load the first frame or a representative image of the court
# Load the image
img_path = 'court1.png'
#img = cv2.imread('court1.png', 1)

# Call the function and get the points
points = click_on_image(img_path)

print('Final list of points:', points)  # The four corner points
"""
"""

import cv2
from event_handler import click_on_image
import matplotlib.path as mplPath
from ball_tracking import *

# Open video file
video_path = '1_1.mp4'
cap = cv2.VideoCapture(video_path)
model_path='/Users/liberin/Desktop/pickleball/best.pt'

# Check if video file was opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Read the first frame
ret, frame = cap.read()

# If frame was read successfully, pass it to the function
if ret:
    points = click_on_image(frame)
    print('Final list of points:', points)
    poly_path = mplPath.Path(points)  # The four corner points
    print(poly_path)
   


    # Play the rest of the video
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Display the resulting frame
        cv2.imshow('Frame', frame)


        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
"""

import cv2
from event_handler import click_on_image
import matplotlib.path as mplPath
from ball_tracking import *

# Open video file
video_path = '1_1.mp4'
cap = cv2.VideoCapture(video_path)
model_path='/Users/liberin/Desktop/pickleball/best.pt'
coordinates=False

# Check if video file was opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Read the first frame
ret, frame = cap.read()

# If frame was read successfully, pass it to the function
if ret:
    print(ret)
    points = click_on_image(frame)
    print('Final list of points:', points)
   

    while True:
        # Capture frame-by-frame
            tracker = BallTracker(model_path, video_path,points)
            tracker.track()
            
           
      