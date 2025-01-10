from screeninfo import get_monitors
import cv2
import numpy as np

def keys_detection(keys, show_screen):
    cap = cv2.VideoCapture(0)  # Open the camera

    # Set the resolution of the camera
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Double the width while keeping the original height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width * 4)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)

    # Print the new resolution for debugging
    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {new_width}x{new_height}")

    prev_gray = None
    _, frame_test = cap.read()
    Y, X, _ = frame_test.shape  # Get the frame dimensions

    # Player
    # Define regions of interest (ROI) for hand detection
    roi_player_width = int(X * 0.15)  # Width of ROI = 15% of the total width
    roi_player_height = int(Y * 0.6)  # Height of ROI = 60% of the total height
    roi_player_top = int(Y * 0.2)  # Distance from the top = 20% of the total height

    # Right-hand ROI
    right_player_hand_roi = (
        int(X * 0.3),  # Start 30% of the way from the left
        roi_player_top,       # Top position
        roi_player_width,     # Width
        roi_player_height     # Height
    )

    # Left-hand ROI
    left_player_hand_roi = (
        int(X * 0.05),  # Start 5% of the way from the left
        roi_player_top,       # Top position
        roi_player_width,     # Width
        roi_player_height     # Height
    )

    # Opponent
    # Define regions of interest (ROI) for hand detection
    roi_Opponent_width = int(X * 0.15)  # Width of ROI = 15% of the total width
    roi_Opponent_height = int(Y * 0.6)  # Height of ROI = 60% of the total height
    roi_Opponent_top = int(Y * 0.2)  # Distance from the top = 20% of the total height

    # Right-hand ROI
    right_Opponent_hand_roi = (
        int(X * 0.8),  # Start 80% of the way from the left
        roi_Opponent_top,       # Top position
        roi_Opponent_width,     # Width
        roi_Opponent_height     # Height
    )

    # Left-hand ROI
    left_Opponent_hand_roi = (
        int(X * 0.55),  # Start 55% of the way from the left
        roi_Opponent_top,       # Top position
        roi_Opponent_width,     # Width
        roi_Opponent_height     # Height
    )

    monitor = get_monitors()[0]  # Use the primary monitor
    screen_width = monitor.width
    screen_height = monitor.height

    scale = 0.25  # Downscaling factor to reduce frame size and processing load

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Split the frame into two halves
        mid_x = frame.shape[1] // 2
        left_frame = frame[:, :mid_x]  # Left half
        right_frame = frame[:, mid_x:]  # Right half

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        small_gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Initialize the previous frame if it's the first iteration
        if prev_gray is None:
            prev_gray = small_gray_frame
            continue

        # Calculate optical flow between the previous and current frames
        flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray_frame, None, 0.5, 2, 9, 2, 5, 1.2, 0)

        # Detect right hand movement
        x, y, w, h = int(right_player_hand_roi[0] * scale), int(right_player_hand_roi[1] * scale), int(right_player_hand_roi[2] * scale), int(right_player_hand_roi[3] * scale)
        right_flow_y = flow[y:y + h, x:x + w, 1]  # Movement along the Y-axis
        keys["player right"] = np.mean(right_flow_y) < -2  # Check upward motion

        # Detect left hand movement
        x, y, w, h = int(left_player_hand_roi[0] * scale), int(left_player_hand_roi[1] * scale), int(left_player_hand_roi[2] * scale), int(left_player_hand_roi[3] * scale)
        left_flow_y = flow[y:y + h, x:x + w, 1]  # Movement along the Y-axis
        keys["player left"] = np.mean(left_flow_y) < -2  # Check upward motion

        # Detect right hand movement
        x, y, w, h = int(right_Opponent_hand_roi[0] * scale), int(right_Opponent_hand_roi[1] * scale), int(right_Opponent_hand_roi[2] * scale), int(right_Opponent_hand_roi[3] * scale)
        right_flow_y = flow[y:y + h, x:x + w, 1]  # Movement along the Y-axis
        keys["opponent right"] = np.mean(right_flow_y) < -2  # Check upward motion

        # Detect left hand movement
        x, y, w, h = int(left_Opponent_hand_roi[0] * scale), int(left_Opponent_hand_roi[1] * scale), int(left_Opponent_hand_roi[2] * scale), int(left_Opponent_hand_roi[3] * scale)
        left_flow_y = flow[y:y + h, x:x + w, 1]  # Movement along the Y-axis
        keys["opponent left"] = np.mean(left_flow_y) < -2  # Check upward motion

        # Update the previous frame
        prev_gray = small_gray_frame

        # Show the two halves of the frame in separate windows
        cv2.imshow('Left Half - Hand Detection', left_frame)
        cv2.moveWindow('Left Half - Hand Detection', int((screen_width - show_screen["game_width"] - X*0.6)/4), int((screen_height - Y)/2))
        cv2.imshow('Right Half - Hand Detection', right_frame)
        cv2.moveWindow('Right Half - Hand Detection', int((3*screen_width - show_screen["game_width"] - X*0.6)/4), int((screen_height - Y)/2))

        # Update the shared state to indicate the camera is visible
        show_screen["show_screen"] = True

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

