import cv2
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.path as mplPath

class BallTracker:
    def __init__(self, model_path, video_path,points):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.points=points
        
        self.kf = self.initialize_kalman_filter()

    @staticmethod
    def initialize_kalman_filter():
        dt = 1.0
        kf = KalmanFilter(dim_x=6, dim_z=2)
        kf.x = np.array([0, 0, 0, 0, 0, 0]) # initial state estimate
        kf.P = np.eye(6) * 1000 # initial error covariance matrix
        kf.F = np.array([[1, 0, dt, 0, 0.5 * (dt ** 2), 0],
                         [0, 1, 0, dt, 0, 0.5 * (dt ** 2)],
                         [0, 0, 1, 0, dt, 0],
                         [0, 0, 0, 1, 0, dt],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])  # state transition matrix
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0]]) # measurement matrix
        kf.R = np.diag([0.1, 0.1]) # measurement noise covariance matrix
        kf.Q = np.array([[dt**4/4, 0, dt**3/2, 0, dt**2, 0],
                         [0, dt**4/4, 0, dt**3/2, 0, dt**2],
                         [dt**3/2, 0, dt**2, 0, dt, 0],
                         [0, dt**3/2, 0, dt**2, 0, dt],
                         [dt**2, 0, dt, 0, 1, 0],
                         [0, dt**2, 0, dt, 0, 1]]) # process noise covariance matrix
        return kf

    def track(self):
        frame_num = 0
        predicted_points = []
        bounce_detected = False
        last_bounce_frame = -10
        test_df = pd.DataFrame(columns=['frame', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'V'])
        detected=False
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

        # rest of your tracking code here
        while True:
            ret, frame = self.cap.read()
            
            """
            k = cv2.waitKey(100)
            fps_in = self.cap.get(cv2.CAP_PROP_FPS)
            fps_out = 0.25
            index_in = -1
            index_out = -1
            success = self.cap.grab()
            if not success: break
            index_in += 1


            out_due = int(index_in / fps_in * fps_out)
            print(out_due)
            if out_due > index_out:
                success, frame = self.cap.retrieve()
                if not success: break
                index_out += 1
            """
                

            #print(fps_in)
            if ret is False:
                break
            bbox = self.model(frame, show=False)
            frame_num += 1
            for boxes_1 in bbox:
                result = boxes_1.boxes.xyxy
                if len(result) == 0 and detected==False:
                    print("not detected")
                else:
                    detected=True
                    
                    if len(result) !=0:
                        cx = int((result[0][0] + result[0][2]) / 2)
                        cy = int((result[0][1] + result[0][3]) / 2)
                        centroid = np.array([cx, cy])
                        self.kf.predict()
                        self.kf.update(centroid)
                    else:
                        cx=int(next_point[0])
                        cy=int(next_point[1])
                        centroid = np.array([cx, cy])
                        self.kf.predict()
                        self.kf.update(centroid)


                    next_point = (self.kf.x).tolist()
                    #predicted_velocity.append((int(next_point[2]),int(next_point[3])))
                    predicted_points.append((int(next_point[0]), int(next_point[1])))
                    if len(predicted_points) > 5:
                        predicted_points.pop(0)
                    #print("next_point", next_point)
                    #print("frame_number", frame_num)
                    if(next_point[2]>0):
                        vx="positive"
                    else:
                        vx="negative"
                    if(next_point[3]>0):
                        vy="positive"
                    else:
                        vy="negative"
                    test_df = test_df._append( { 'frame': frame_num, 'x': next_point[0], 'y': next_point[1], 'vx': next_point[2],'vy':next_point[3]
                                            ,'ax':next_point[4],'ay':next_point[5],'V': np.sqrt(self.kf.x[2]**2 + self.kf.x[3]**2)}, ignore_index=True)
                    
                    cv2.putText(frame, f'Frame: {frame_num}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #cv2.putText(frame, f': {next_point}', (10,205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    #cv2.putText(frame, f'vx:{vx}',(10,205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    #cv2.putText(frame, f'vy:{vy}',(10,230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0,0,255), 5)
                    cv2.circle(frame, (int(next_point[0]), int(next_point[1])), 5, (255, 0, 0), 10)
                    for i, p in enumerate(predicted_points):
                        color = (255,255,255)
                        cv2.circle(frame, p, 5, color, 2)
                    #if (test_df.shape[0] > 1 and test_df.shape[1] > 3):
                        #print(frame_num)
                        #print(test_df.iloc[-2, 2])
                        #print((self.kf.x[2])**2)
                        #print(np.sqrt((test_df.iloc[-2, 2])-((self.kf.x[2])**2)))
                    if not bounce_detected and frame_num - last_bounce_frame > 20:
                        if ((test_df.shape[0] > 1 and test_df.shape[1] > 3) and (round(test_df.iloc[-2, 7])== round(np.sqrt(self.kf.x[2]**2 + self.kf.x[3]**2))) or ((round(test_df.iloc[-2, 7])) - round(np.sqrt(self.kf.x[2]**2 + self.kf.x[3]**2)) ==1 )):
                           # print(test_df.iloc[-2, 3])
                            #print(test_df.iloc[-2, 3])
                            bounce_detected = True
                            last_bounce_frame = frame_num
                            print("Bounce detected")
                        #if ((test_df.shape[0] > 1 and test_df.shape[1] > 3 and np.sign(test_df.iloc[-2, 3]) == np.sign(kf.x[2])) and (test_df.shape[0] > 1 and test_df.shape[1] > 3 and np.sign(test_df.iloc[-2, 4]) > 0 and np.sign(kf.x[3]) < 0 and np.sqrt((test_df.iloc[-2, 2])-((kf.x[2])**2))<25)):
                        #if kf.x[3]< 0 and kf.x[1] <= 0.3048:# If Y acceleration is less than the negative threshold, say -15
                        #or (round(test_df.iloc[-2, 7])- round(np.sqrt(kf.x[2]**2 + kf.x[3]**2)) <1 ))
                        
                            
                        """
                    if not bounce_detected and frame_num - last_bounce_frame > 10:
                        if kf.x[2] < 0 and kf.x[3]: # If Y acceleration is less than the negative threshold, say -15
                            bounce_detected = True
                            last_bounce_frame = frame_num
                            print("Bounce detected")
                            """
                    if bounce_detected:
                        cv2.putText(frame, 'Bounce Detected', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        poly_path = mplPath.Path(self.points)  # The four corner points
                        print(poly_path)
                        ball_x_pred,ball_y_pred=next_point[0],next_point[1]
                        pred_ball_centroid = (ball_x_pred, ball_y_pred)
                        if poly_path.contains_point(pred_ball_centroid):
                            print("The ball is in.")
                            cv2.putText(frame, 'Ball is in', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            print("The ball is out.")
                            cv2.putText(frame, 'Ball is in', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if self.kf.x[5] > 0: # After a bounce is detected, wait until acceleration is above the threshold, say -5, to detect the bounce again
                        bounce_detected = False
                   # print(test_df)
                    test_df.to_csv('file.csv')
                    
                    cv2.imshow('raw', frame)
                    
                    # Uncomment the following lines to save the output video
                    out.write(frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
       
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


        


# use the BallTracker class:
#model_path = '/Users/liberin/Desktop/pickleball/best.pt'
#video_path = '1_1.mp4'
#tracker = BallTracker(model_path, video_path)
#tracker.track()
