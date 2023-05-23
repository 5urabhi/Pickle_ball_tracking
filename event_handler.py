"""
#working for video
import cv2

def click_on_image(image):
    # Initial variables
    points = []
    done = False

    # Click event handler
    def click_event(event, x, y, flags, param):
        nonlocal done
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            points.append((x, y))
            print(f'Point added: {x, y}')
            if len(points) >= 4:
                print('Four points added, closing window...')
                done = True

    # Load image
    #image = cv2.imread(image)
    cv2.imshow('image', image)

    # Set mouse callback
    cv2.setMouseCallback('image', click_event)

    # Keep window open until 4 points have been clicked or 'q' has been pressed
    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q') or done:
            break

    # Clean up
    cv2.destroyAllWindows()

    return points

# Use the function
#points = click_on_image('court1.png')
#print('Final list of points:', points)
"""

"""
# event_handler.py
# event_handler.py
import cv2

def click_on_image(image):
    # Initial variables
    points = []
    done = False
    def click_event(event, x, y, flags, param):
        nonlocal done
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            points.append((x, y))
            print(f'Point added: {x, y}')
            if len(points) >= 4:
                print('Four points added, closing window...')
                cv2.destroyAllWindows()

    # Load image
    image = cv2.imread(image)
    cv2.imshow('image', image)

    # Set mouse callback
    cv2.setMouseCallback('image', click_event)

    # Keep window open until 4 points have been clicked or 'q' has been pressed
    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q') or done:
            break

    # Clean up
    cv2.destroyAllWindows()

    return points

"""
import cv2
# coordinates.py
def click_on_image(image):
    # Initial variables
    points = []
    done = False

    # Click event handler
    def click_event(event, x, y, flags, param):
        nonlocal done
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            points.append((x, y))
            print(f'Point added: {x, y}')
            if len(points) >= 4:
                print('Four points added, closing window...')
                done = True

    # Load image
    cv2.imshow('image', image)

    # Set mouse callback
    cv2.setMouseCallback('image', click_event)

    # Keep window open until 4 points have been clicked or 'q' has been pressed
    while not done:
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()

    return points
