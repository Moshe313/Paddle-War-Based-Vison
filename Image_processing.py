import cv2
import numpy as np
from screeninfo import get_monitors

def draw_semitransparent_rectangle(frame, x, y, w, h, color=(0, 0, 0), alpha=0.5):
    """
    Draws a semi-transparent rectangle over the region (x, y, w, h) in 'frame'.
    'color' is in BGR format. 'alpha' is the opacity for the rectangle [0..1].
    """
    # Create an overlay from the original frame
    overlay = frame.copy()

    # Draw a filled rectangle on the overlay
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

    # Blend the overlay with the original frame using alpha
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def keys_detection(keys, show_screen):
    cap = cv2.VideoCapture(0)  # Open the camera

    # Set the resolution to triple the original width
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width * 3)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {new_width}x{new_height}")

    ret, frame_test = cap.read()
    if not ret:
        print("Failed to read from camera.")
        return
    Y, X, _ = frame_test.shape

    # Define player hand ROIs
    roi_player_width = int(X * 0.15)
    roi_player_height = int(Y * 0.6)
    roi_player_top = int(Y * 0.2)
    right_player_hand_roi = (int(X * 0.3), roi_player_top, roi_player_width, roi_player_height)
    left_player_hand_roi = (int(X * 0.05), roi_player_top, roi_player_width, roi_player_height)

    # Define opponent hand ROIs
    roi_opponent_width = int(X * 0.15)
    roi_opponent_height = int(Y * 0.6)
    roi_opponent_top = int(Y * 0.2)
    right_opponent_hand_roi = (int(X * 0.8), roi_opponent_top, roi_opponent_width, roi_opponent_height)
    left_opponent_hand_roi = (int(X * 0.55), roi_opponent_top, roi_opponent_width, roi_opponent_height)

    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    scale = 0.25  # Downscale factor for processing
    show_screen["left_cam"] = None
    show_screen["right_cam"] = None

    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        # --- Draw semi-transparent rectangles for each ROI ---
        # Player Right Hand ROI
        x, y, w, h = right_player_hand_roi
        draw_semitransparent_rectangle(frame, x, y, w, h, color=(0, 0, 0), alpha=0.3)

        # Player Left Hand ROI
        x, y, w, h = left_player_hand_roi
        draw_semitransparent_rectangle(frame, x, y, w, h, color=(0, 0, 0), alpha=0.3)

        # Opponent Right Hand ROI
        x, y, w, h = right_opponent_hand_roi
        draw_semitransparent_rectangle(frame, x, y, w, h, color=(0, 0, 0), alpha=0.3)

        # Opponent Left Hand ROI
        x, y, w, h = left_opponent_hand_roi
        draw_semitransparent_rectangle(frame, x, y, w, h, color=(0, 0, 0), alpha=0.3)
        # -----------------------------------------------------

        # Downscale and convert to grayscale for optical flow
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        small_gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = small_gray_frame
            # Split the frame into left and right for display
            mid_x = frame.shape[1] // 2
            show_screen["left_cam"] = frame[:, :mid_x]
            show_screen["right_cam"] = frame[:, mid_x:]
            continue

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray_frame, None,
                                            0.5, 2, 9, 2, 5, 1.2, 0)

        # Player Right
        x, y, w, h = [int(v * scale) for v in right_player_hand_roi]
        right_flow_y = flow[y:y+h, x:x+w, 1]
        keys["player right"] = np.mean(right_flow_y) < -2

        # Player Left
        x, y, w, h = [int(v * scale) for v in left_player_hand_roi]
        left_flow_y = flow[y:y+h, x:x+w, 1]
        keys["player left"] = np.mean(left_flow_y) < -2

        # Opponent Right
        x, y, w, h = [int(v * scale) for v in right_opponent_hand_roi]
        right_flow_y = flow[y:y+h, x:x+w, 1]
        keys["opponent right"] = np.mean(right_flow_y) < -2

        # Opponent Left
        x, y, w, h = [int(v * scale) for v in left_opponent_hand_roi]
        left_flow_y = flow[y:y+h, x:x+w, 1]
        keys["opponent left"] = np.mean(left_flow_y) < -2

        prev_gray = small_gray_frame

        # If requested, capture and save images with hand overlays
        if show_screen["pic"]:
            left_hand = cv2.imread('left_hand.png')
            right_hand = cv2.imread('right_hand.png')
            mid_x = frame.shape[1] // 2
            left_frame = frame[:, :mid_x]
            right_frame = frame[:, mid_x:]

            target_height = right_frame.shape[0]
            l_scaling_factor = target_height / left_hand.shape[0]
            r_scaling_factor = target_height / right_hand.shape[0]
            l_new_width = int(left_hand.shape[1] * l_scaling_factor)
            r_new_width = int(right_hand.shape[1] * r_scaling_factor)
            l_resized_hand = cv2.resize(left_hand, (l_new_width, target_height), interpolation=cv2.INTER_LINEAR)
            r_resized_hand = cv2.resize(right_hand, (r_new_width, target_height), interpolation=cv2.INTER_LINEAR)

            l_combined_image = np.hstack((l_resized_hand, left_frame))
            r_combined_image = np.hstack((right_frame, r_resized_hand))
            cv2.imwrite('left_frame.jpg', l_combined_image)
            print("Left frame saved as 'left_frame.jpg'")
            cv2.imwrite('right_frame.jpg', r_combined_image)
            print("Right frame saved as 'right_frame.jpg'")
            show_screen["pic"] = False

        # Finally, split the frame to update show_screen for the rest of the program
        mid_x = frame.shape[1] // 2
        show_screen["left_cam"] = frame[:, :mid_x]
        show_screen["right_cam"] = frame[:, mid_x:]
        show_screen["video_prepared"] = True
        show_screen["show_screen"] = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()