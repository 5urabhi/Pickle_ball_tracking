# Pickleball Tracking System

## Overview

This project utilizes advanced computer vision techniques to track a pickleball during gameplay, determining both the bounce status of the ball (bounced or not) and its position relative to the court boundaries (in or out). It employs a combination of libraries including OpenCV, Matplotlib, Ultralytics, NumPy, Pandas, and FilterPy to analyze video footage of pickleball games, providing valuable insights for players, coaches, and enthusiasts.

## Installation

Before running the code, ensure you have Python installed on your system. This project has been tested on Python 3.8+. You will need to install several third-party libraries to get started.

### Required Libraries

- OpenCV
- Matplotlib
- Ultralytics
- NumPy
- Pandas
- FilterPy

You can install all required libraries using pip:

```
pip install opencv-python-headless matplotlib ultralytics numpy pandas filterpy
```
Note: We use opencv-python-headless to avoid unnecessary GUI dependencies for server environments.

## Usage
To use this project, follow these steps:

### Prepare Your Data: 
Ensure you have video footage of pickleball games saved in a supported format (e.g., MP4, AVI).

### Run the Tracker: 
Execute the main script (ball_in_out.py). Click the four corners of the court when the video starts.

### Review the Results: 
The script will process the video, tracking the pickleball throughout its trajectory. The results, including bounce status and in/out position, will be saved in a specified output directory.

## Interpreting Results



https://github.com/5urabhi/Pickle_ball/assets/104481755/e12be456-8517-4f17-b295-0ddac2e2562d



## Contributing
Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT

