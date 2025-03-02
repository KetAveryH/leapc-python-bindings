import json
import time
import os
import ctypes
from collections import deque
import numpy as np
import math

def normalize(x, y, z):
    """
    Returns a normalized (unit) vector from (x, y, z).
    If the magnitude is extremely small, return (0, 0, 0) to avoid division by zero.
    """
    mag = math.sqrt(x*x + y*y + z*z)
    if mag < 1e-8:
        return (0.0, 0.0, 0.0)
    return (x/mag, y/mag, z/mag)



class HandTrackerBuffer:
    def __init__(self, maxlen=10):
        # We'll keep up to 10 frames of data
        self.data_buffer = deque(maxlen=maxlen)
    
    def append(self, event):
        """
        event: an object that has event.hands,
            each hand has .type and .palm.position.x, etc.
        """
        left_coords = None
        right_coords = None
        
        for hand in event.hands:
            hand_type = "left" if hand.type.value == 0 else "right"
            x = hand.palm.position.x
            y = hand.palm.position.y
            z = hand.palm.position.z
            
            if hand_type == "left":
                left_coords = (x, y, z)
            else:
                right_coords = (x, y, z)

        entry = {
            "timestamp": time.time(),
            "left":  left_coords,
            "right": right_coords
        }

        # Store in the deque
        self.data_buffer.append(entry)

    def calculate_velocity(self, hand_type='left'):
        """
        Average velocity by summing velocities between consecutive frames
        and dividing by the number of intervals.
        """
        if len(self.data_buffer) < 2:
            return (0.0, 0.0, 0.0)

        total_vx = 0.0
        total_vy = 0.0
        total_vz = 0.0
        valid_pairs = 0

        for i in range(len(self.data_buffer) - 1):
            current = self.data_buffer[i]
            nxt     = self.data_buffer[i + 1]

            # Skip if missing data
            if current[hand_type] is None or nxt[hand_type] is None:
                continue

            (cx, cy, cz) = current[hand_type]
            (nx, ny, nz) = nxt[hand_type]

            dt = nxt["timestamp"] - current["timestamp"]
            if dt <= 0:
                continue

            vx = (nx - cx) / dt
            vy = (ny - cy) / dt
            vz = (nz - cz) / dt

            total_vx += vx
            total_vy += vy
            total_vz += vz
            valid_pairs += 1

        if valid_pairs == 0:
            return (0.0, 0.0, 0.0)

        return (total_vx / valid_pairs,
                total_vy / valid_pairs,
                total_vz / valid_pairs)


class ActionController:
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP   = 0x0004
    SWIPE_THRESHOLD = 650

    def __init__(self, canvas, enable_control=False):
        """
        :param canvas: The Canvas instance used for drawing.
        :param enable_control: If False, disable ALL OS-level actions
                               (mouse movement, click, scroll).
        """
        self.canvas = canvas
        self.enable_control = enable_control

        self.hand_state = {"left": "idle", "right": "idle"}
        self.complex_state = "idle"
        self.hand_press_time = {"left": 0.0, "right": 0.0}
        self.curr_hand = None

        # Track last time we updated a complex gesture (e.g., swipe)
        self.complex_gesture_timestamp = 0.0
        
        self.hand_buffer = HandTrackerBuffer(maxlen=10)
        self.right_hand_velocity = (0.0, 0.0, 0.0)
        self.left_hand_velocity  = (0.0, 0.0, 0.0)
        
        # Pinch/Grab thresholds
        self.pinch_threshold = 0.8
        self.grab_threshold  = 0.9
        self.hold_threshold  = 0.1

        # For "grab to scroll"
        self.scroll_sensitivity = 0.8
        self.last_scroll_y = 0.0

        # Play area
        self.max_min_x = [260, -180]
        self.max_min_y = [50, 0]
        self.max_min_z = [290, -140]

        # Capture device dimensions
        self.cap_width = self.max_min_x[0] - self.max_min_x[1]
        self.cap_height = self.max_min_y[0] - self.max_min_y[1]

        # Screen resolution
        self.screen_width  = 1920
        self.screen_height = 1080

        self.pinch_timer = 0.0  # Timer to track time since last pinch
        self.pinch_timeout = 0.1  # Time frame to consider a swipe after pinch (in seconds)
        self.left_pinch_time = 0.0  # Track the last pinch times separately
        self.right_pinch_time = 0.0  # Track the last pinch times separately
        self.last_pinch_time = 0.0  # Last time a pinch was detected

        self.gesture_timeout = 0.3  # Time frame to reset complex gesture (in seconds)
        self.last_gesture_time = 0.0  # Last time a gesture was observed

    # -------------------------------------------------------------------------
    #  get_state() -> JSON with positional, gesture, and timestamp info
    # -------------------------------------------------------------------------
    def get_state(self):
        """
        Return the structured JSON:
        {
          "left_hand": {
            "position": {"x":..., "y":..., "z":...},
            "gesture": "...",
            "timestamp": ...
          },
          "right_hand": {
            "position": {"x":..., "y":..., "z":...},
            "gesture": "...",
            "timestamp": ...
          },
          "complex_gesture": {
            "gesture": "...",
            "gesture_timestamp": ...
          }
        }
        
        if complex_gesture == "zoom":
        "complex_gesture": {
        "gesture": "zoom",
        "gesture_timestamp": 1691858647.1234,
        "zoom": {
            "multiplier": 1.15
            }
        }

        """
        
        left_coords = None
        right_coords = None
        left_timestamp = None
        right_timestamp = None

        if self.hand_buffer.data_buffer:
            latest = self.hand_buffer.data_buffer[-1]
            left_coords = latest["left"]
            right_coords = latest["right"]
            frame_timestamp = latest["timestamp"]

            if left_coords is not None:
                left_timestamp = frame_timestamp
            if right_coords is not None:
                right_timestamp = frame_timestamp

        state_dict = {
            "left_hand": {
                "position": {
                    "x": left_coords[0],
                    "y": left_coords[1],
                    "z": left_coords[2]
                } if left_coords else None,
                "gesture": self.hand_state["left"],
                "timestamp": left_timestamp
            },
            "right_hand": {
                "position": {
                    "x": right_coords[0],
                    "y": right_coords[1],
                    "z": right_coords[2]
                } if right_coords else None,
                "gesture": self.hand_state["right"],
                "timestamp": right_timestamp
            },
            "complex_gesture": {
                "gesture": self.complex_state,
                "gesture_timestamp": self.complex_gesture_timestamp
            }
        }

        # If we are in a zoom state, embed a "zoom" object
        if self.complex_state == "zoom":
            state_dict["complex_gesture"]["zoom"] = {
                "multiplier": self.zoom_multiplier
            }

        return json.dumps(state_dict, indent=2)

    # --------------------
    # Configuration
    # --------------------
    def save_config(self, filename="play_area_config.json"):
        config = {
            "max_min_x": self.max_min_x,
            "max_min_y": self.max_min_y,
            "max_min_z": self.max_min_z
        }
        with open(filename, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {filename}")

    def load_config(self, filename="play_area_config.json"):
        if not os.path.exists(filename):
            print(f"No config file found at {filename}. Using defaults.")
            return

        with open(filename, "r") as f:
            config = json.load(f)

        self.max_min_x = config.get("max_min_x", self.max_min_x)
        self.max_min_y = config.get("max_min_y", self.max_min_y)
        self.max_min_z = config.get("max_min_z", self.max_min_z)

        print(f"Configuration loaded from {filename}")

    def reset_setup(self):
        self.max_min_x = [260, -180]
        self.max_min_y = [50, 0]
        self.max_min_z = [290, -140]

    # --------------------
    # Event Router
    # --------------------
    def tracking_event_router(self, event, state):
        """
        States:
          0 -> Sleep
          1 -> Active
          2 -> Setup
        """
        if state == 0:
            # Sleep state => do nothing
            pass
        elif state == 1:
            self.active_handler(event)
        elif state == 2:
            self.setup_handler(event)
        else:
            raise ValueError(f"Unexpected state: {state}")

    # --------------------
    # Active mode
    # --------------------
    def active_handler(self, event):
        """Process gestures and optionally move mouse, etc."""
        self.hand_buffer.append(event)

        self.right_hand_velocity = self.hand_buffer.calculate_velocity('right')
        self.left_hand_velocity  = self.hand_buffer.calculate_velocity('left')

        present_hands = set()  # track which hands are actually present

        for hand in event.hands:
            hand_type = "left" if hand.type.value == 0 else "right"
            present_hands.add(hand_type)
            self.curr_hand = hand
            
            # For determining hand orientation
            device_forward = np.array([0, 0, 1])  # Leap Motion is typically +Z
            palm_normal = np.array([hand.palm.normal.x, hand.palm.normal.y, hand.palm.normal.z])
            dot_product = np.dot(palm_normal, device_forward)  # if negative palm is facing device, positive is facing away
            
            if hand_type == "right":
                px, py = hand.palm.position.x, hand.palm.position.y
                self.move_cursor(px, py, 1, 1)  # Only moves mouse if enable_control = True
                self.update_right_hand_state(hand.grab_strength, hand.pinch_strength, hand.palm.position.y, dot_product)

            elif hand_type == "left":
                self.update_left_hand_state(hand.grab_strength, hand.pinch_strength, hand.palm.position.y, dot_product)

        # Check for swipe after pinch for **both hands**
        self._check_for_pinch_swipe("right")
        self._check_for_pinch_swipe("left")

        # Check if index fingers are overlapping or crossed
        crossed_finger_status = self.check_finger_cross(event.hands)

        if crossed_finger_status["overlapping"]:
            # print("[Gesture] Index fingers are overlapping!")  # Removed print statement
            if crossed_finger_status["crossed"]:
                # print("[Gesture] Index fingers are crossed!")  # Removed print statement
                self.complex_state = "finger-cross"  # Set complex state to finger-cross

        recognized_gesture = self.canvas.get_and_forget_drawn_gesture()
        
        if recognized_gesture:
            print(f"[ActionController] Detected new drawn gesture: {recognized_gesture}")
            self.complex_state = recognized_gesture
            self.complex_gesture_timestamp = time.time()
            self.last_gesture_time = time.time()  # Update last gesture time
        
        # ---- AFTER processing all hands ----
        # If the right hand is missing but we are stuck on a swipe, reset to idle
        if "right" not in present_hands and self.complex_state != "idle":
            # print("[Controller] Right hand lost => resetting complex gesture to idle.")
            self.complex_state = "idle"
            self.complex_gesture_timestamp = time.time()   
        
        # Reset complex gesture if no gestures have been observed for a while
        current_time = time.time()
        if current_time - self.last_gesture_time > self.gesture_timeout:
            self.complex_state = "idle"
            self.last_gesture_time = current_time  # Reset the last gesture time
        
        # -----------------------------------------------------
        # ZOOM LOGIC: Check if both hands are in a grab-holding
        # -----------------------------------------------------
        left_st  = self.hand_state["left"]
        right_st = self.hand_state["right"]

        # Some people might define 'grab_holding' states as e.g. 
        #    "grab-away-holding" OR "grab-towards-holding"
        # Adjust this check to match your naming:
        left_is_grabbing  = (left_st  in ["grab-away-holding", "grab-towards-holding"])
        right_is_grabbing = (right_st in ["grab-away-holding", "grab-towards-holding"])

        if left_is_grabbing and right_is_grabbing:
            dist = self.distance_between_hands()
            if dist is not None:
                # If we haven't established a baseline, set it now
                if self.zoom_baseline_distance is None:
                    self.zoom_baseline_distance = dist
                    self.zoom_multiplier = 1.0
                    self.complex_state = "zoom"
                    self.complex_gesture_timestamp = time.time()
                else:
                    # compute ratio
                    if self.zoom_baseline_distance > 1e-6:
                        self.zoom_multiplier = dist / self.zoom_baseline_distance
                        # We'll remain in "zoom" as long as both are grabbing
                        if self.complex_state != "zoom":
                            self.complex_state = "zoom"
                            self.complex_gesture_timestamp = time.time()
            # if dist is None, do nothing special
        else:
            # If either hand left the grab-holding state,
            # end the zoom gesture
            if self.complex_state == "zoom":
                self.complex_state = "idle"
                self.complex_gesture_timestamp = time.time()
            
            # Reset the baseline
            self.zoom_baseline_distance = None
            self.zoom_multiplier = 1.0
             
    def _check_for_pinch_swipe(self, hand_type):
        """
        Checks for swipe gestures while pinch-holding or after pinch is released.
        This function is now hand-specific.
        """
        current_time = time.time()
        pinch_time = self.left_pinch_time if hand_type == "left" else self.right_pinch_time
        time_since_last_pinch = current_time - pinch_time

        if time_since_last_pinch <= self.pinch_timeout:
            velocity = self.left_hand_velocity if hand_type == "left" else self.right_hand_velocity
            vx, vy, vz = velocity

            # Detect swipes based on velocity
            if abs(vx) > self.SWIPE_THRESHOLD or abs(vy) > self.SWIPE_THRESHOLD:
                if vx > self.SWIPE_THRESHOLD:
                    self.hand_state[hand_type] = "swipe-right"
                elif vx < -self.SWIPE_THRESHOLD:
                    self.hand_state[hand_type] = "swipe-left"
                elif vy > self.SWIPE_THRESHOLD:
                    self.hand_state[hand_type] = "swipe-up"
                elif vy < -self.SWIPE_THRESHOLD:
                    self.hand_state[hand_type] = "swipe-down"

    def distance_between_hands(self):
        """
        Returns the 3D distance between left and right palm positions (in mm),
        or None if one of them is missing.
        """
        if not self.hand_buffer.data_buffer:
            return None
        
        latest = self.hand_buffer.data_buffer[-1]
        left_coords = latest["left"]   # (x, y, z) or None
        right_coords = latest["right"] # (x, y, z) or None
        
        if (left_coords is None) or (right_coords is None):
            return None
        
        lx, ly, lz = left_coords
        rx, ry, rz = right_coords
        dx = rx - lx
        dy = ry - ly
        dz = rz - lz
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)

                

    def get_thumb_direction(self, thumb):
        """
        Returns a 3-tuple (tx, ty, tz) indicating the direction 
        from the base of the thumb's metacarpal to the tip of the distal bone, normalized.
        """
        base_x = thumb.metacarpal.prev_joint.x
        base_y = thumb.metacarpal.prev_joint.y
        base_z = thumb.metacarpal.prev_joint.z

        tip_x  = thumb.distal.next_joint.x
        tip_y  = thumb.distal.next_joint.y
        tip_z  = thumb.distal.next_joint.z

        dx = tip_x - base_x
        dy = tip_y - base_y
        dz = tip_z - base_z

        return normalize(dx, dy, dz)



    def classify_thumb_direction(self, thumb):
        """
        Returns "thumb-up" if the thumb direction has a positive Y component above a threshold,
        "thumb-down" if the thumb direction has a negative Y component below a threshold,
        otherwise returns None (too horizontal, etc.).
        """
        tx, ty, tz = self.get_thumb_direction(thumb)
        # Dot product with (0,1,0) is just 'ty'

        # Tune thresholds as needed:
        if ty > 0.5:
            return "thumb-up"
        elif ty < -0.5:
            return "thumb-down"
        else:
            return None

    def check_thumb_gesture(self, hand):
        """
        Returns:
        "thumbs_up"   if only the thumb is extended and it's pointing up,
        "thumbs_down" if only the thumb is extended and it's pointing down,
        None          otherwise.
        """
        thumb  = hand.thumb
        index  = hand.index
        middle = hand.middle
        ring   = hand.ring
        pinky  = hand.pinky

        # 1) Check extension: only the thumb is extended
        if thumb.is_extended and not any(f.is_extended for f in [index, middle, ring, pinky]):
            # 2) Check direction
            direction = self.classify_thumb_direction(thumb)  # => "thumb-up", "thumb-down", or None
            if direction == "thumb-up":
                return "thumbs_up"
            elif direction == "thumb-down":
                return "thumbs_down"

        return None

    def check_finger_cross(self, hands, overlap_threshold=40.0, orthogonal_threshold=0.35):
        """
        Checks if the index fingers of both hands are overlapping and/or crossed.
        """
        left_hand = None
        right_hand = None

        # Ensure only two hands are being processed
        if len(hands) > 2:
            raise Exception("Too many hands detected! Limit detection to two.")

        # Identify left and right hands
        for hand in hands:
            if hand.type.value == 0:
                left_hand = hand
            else:
                right_hand = hand

        # If either hand is missing, we cannot check
        if not left_hand or not right_hand:
            return {"overlapping": False, "crossed": False}

        # ---- Overlapping Check ----
        left_middle_tip = left_hand.index.proximal.next_joint
        right_middle_tip = right_hand.index.proximal.next_joint

        # Compute Euclidean distance between tips of fingers
        distance = math.sqrt(
            (left_middle_tip.x - right_middle_tip.x) ** 2 +
            (left_middle_tip.y - right_middle_tip.y) ** 2 +
            (left_middle_tip.z - right_middle_tip.z) ** 2
        )

        are_overlapping = distance < overlap_threshold

        # ---- Orthogonality Check ----
        left_dir = (
            left_hand.index.distal.next_joint.x - left_hand.index.proximal.prev_joint.x,
            left_hand.index.distal.next_joint.y - left_hand.index.proximal.prev_joint.y,
            left_hand.index.distal.next_joint.z - left_hand.index.proximal.prev_joint.z,
        )

        right_dir = (
            right_hand.index.distal.next_joint.x - right_hand.index.proximal.prev_joint.x,
            right_hand.index.distal.next_joint.y - right_hand.index.proximal.prev_joint.y,
            right_hand.index.distal.next_joint.z - right_hand.index.proximal.prev_joint.z,
        )

        # Normalize the vectors
        left_dir = normalize(*left_dir)
        right_dir = normalize(*right_dir)

        # Compute dot product
        dot_product = (
            left_dir[0] * right_dir[0] +
            left_dir[1] * right_dir[1] +
            left_dir[2] * right_dir[2]
        )

        are_crossed = abs(dot_product) < orthogonal_threshold

        return {"overlapping": are_overlapping, "crossed": are_crossed}

    def update_right_hand_state(self, grab_strength, pinch_strength, palm_y, dot_product):
        """
        Extended state machine for the right hand, including:
        - open-palm-away
        - open-palm-towards
        - pinch-pressing / pinch-holding (with swipe detection)
        - grab-away-pressing / grab-away-holding
        - grab-towards-pressing / grab-towards-holding
        - NEW: thumbs_up, thumbs_down
        """
        current_time = time.time()
        current_state = self.hand_state["right"]

        pinch_active = (pinch_strength >= self.pinch_threshold)
        grab_active  = (grab_strength >= self.grab_threshold)

        # dot < 0 => palm_away, dot >= 0 => palm_towards
        palm_orientation = "palm_away" if (dot_product < 0) else "palm_towards"

        # Decide if user is pinching or grabbing:
        gesture = None
        # Check if pinch is active and index finger is not extended
        if pinch_active and (pinch_strength > grab_strength) and not self.curr_hand.index.is_extended:
            gesture = "pinch"
            self.right_pinch_time = current_time  # Update pinch time
        elif grab_active:
            if palm_orientation == "palm_away":
                gesture = "grab_palm_away"
            else:
                gesture = "grab_palm_towards"

        # -------------------------------------------------------------
        #   OPEN-PALM states => default fallback if no pinch/grab
        # -------------------------------------------------------------
        if current_state in ("idle", "open-palm-away", "open-palm-towards"):
            if gesture == "pinch":
                self.hand_state["right"] = "pinch-pressing"
                self.hand_press_time["right"] = current_time

            elif gesture == "grab_palm_away":
                self.hand_state["right"] = "grab-away-pressing"
                self.hand_press_time["right"] = current_time
                self.last_scroll_y = palm_y

            elif gesture == "grab_palm_towards":
                self.hand_state["right"] = "grab-towards-pressing"
                self.hand_press_time["right"] = current_time
                self.last_scroll_y = palm_y

            else:
                # No pinch/grab => remain in open-palm
                if palm_orientation == "palm_away":
                    self.hand_state["right"] = "open-palm-away"
                else:
                    self.hand_state["right"] = "open-palm-towards"

        # -------------------------------------------------------------
        #   PINCH-PRESSING -> pinch-holding or short click
        # -------------------------------------------------------------
        elif current_state == "pinch-pressing":
            if gesture == "pinch":
                elapsed = current_time - self.hand_press_time["right"]
                if elapsed >= self.hold_threshold:
                    self.hand_state["right"] = "pinch-holding"
                    self.press_down()  # e.g. mouse down
            else:
                # short pinch => quick click
                elapsed = current_time - self.hand_press_time["right"]
                if elapsed < self.hold_threshold:
                    self.trigger_click_event()
                # revert
                if palm_orientation == "palm_away":
                    self.hand_state["right"] = "open-palm-away"
                else:
                    self.hand_state["right"] = "open-palm-towards"

        # -------------------------------------------------------------
        #   PINCH-HOLDING -> check swipes, or end pinch
        # -------------------------------------------------------------
        elif current_state == "pinch-holding":
            if gesture == "pinch":
                self._check_for_pinch_swipe("right")  # Check for swipe while pinch-holding
            else:
                # end pinch
                self.press_up()
                if palm_orientation == "palm_away":
                    self.hand_state["right"] = "open-palm-away"
                else:
                    self.hand_state["right"] = "open-palm-towards"
                # If the pinch gesture is released, start the timer
                self.pinch_timer = time.time()  # Reset the timer

        # -------------------------------------------------------------
        #   GRAB-AWAY: pressing -> holding
        # -------------------------------------------------------------
        elif current_state == "grab-away-pressing":
            if gesture == "grab_palm_away":
                elapsed = current_time - self.hand_press_time["right"]
                if elapsed >= self.hold_threshold:
                    self.hand_state["right"] = "grab-away-holding"
                    # e.g. start scrolling
            else:
                if palm_orientation == "palm_away":
                    self.hand_state["right"] = "open-palm-away"
                else:
                    self.hand_state["right"] = "open-palm-towards"

        elif current_state == "grab-away-holding":
            if gesture == "grab_palm_away":
                self.scroll_with_displacement(palm_y)
            else:
                if palm_orientation == "palm_away":
                    self.hand_state["right"] = "open-palm-away"
                else:
                    self.hand_state["right"] = "open-palm-towards"

        # -------------------------------------------------------------
        #   GRAB-TOWARDS: pressing -> holding
        # -------------------------------------------------------------
        elif current_state == "grab-towards-pressing":
            if gesture == "grab_palm_towards":
                elapsed = current_time - self.hand_press_time["right"]
                if elapsed >= self.hold_threshold:
                    self.hand_state["right"] = "grab-towards-holding"
                    # e.g. do something special
            else:
                if palm_orientation == "palm_away":
                    self.hand_state["right"] = "open-palm-away"
                else:
                    self.hand_state["right"] = "open-palm-towards"

        elif current_state == "grab-towards-holding":
            if gesture == "grab_palm_towards":
                self.scroll_with_displacement(palm_y)
            else:
                if palm_orientation == "palm_away":
                    self.hand_state["right"] = "open-palm-away"
                else:
                    self.hand_state["right"] = "open-palm-towards"

        # -------------------------------------------------------------
        #   Unhandled fallback
        # -------------------------------------------------------------
        else:
            if palm_orientation == "palm_away":
                self.hand_state["right"] = "open-palm-away"
            else:
                self.hand_state["right"] = "open-palm-towards"




    def update_left_hand_state(self, grab_strength, pinch_strength, palm_y, dot_product):
        """
        Extended state machine for the left hand, including orientation-based
        open palm states (open-palm-away / open-palm-towards), pinch, grab,
        and thumbs-up/down detection.
        """
        current_time  = time.time()
        current_state = self.hand_state["left"]

        pinch_active = (pinch_strength >= self.pinch_threshold)
        grab_active  = (grab_strength >= self.grab_threshold)

        palm_orientation = "palm_away" if (dot_product < 0) else "palm_towards"
        gesture = None
        # checks if index finger 
        if pinch_active and (pinch_strength > grab_strength) and not self.curr_hand.index.is_extended: # this is_extended was added as a pinch would be registered when making a pointer finger
            gesture = "pinch"
            self.left_pinch_time = current_time  # Update pinch time
        elif grab_active:
            if palm_orientation == "palm_away":
                gesture = "grab_palm_away"
            else:
                gesture = "grab_palm_towards"

        # -------------------------------------------------------------
        #   OPEN-PALM states => default if no pinch/grab
        # -------------------------------------------------------------
        if current_state in ("idle", "open-palm-away", "open-palm-towards"):
            if gesture == "pinch":
                self.hand_state["left"] = "pinch-pressing"
                self.hand_press_time["left"] = current_time
                # e.g., begin drawing
                self.canvas.begin_drawing()

            elif gesture == "grab_palm_away":
                self.hand_state["left"] = "grab-away-pressing"
                self.hand_press_time["left"] = current_time

            elif gesture == "grab_palm_towards":
                self.hand_state["left"] = "grab-towards-pressing"
                self.hand_press_time["left"] = current_time

            else:
                if palm_orientation == "palm_away":
                    self.hand_state["left"] = "open-palm-away"
                else:
                    self.hand_state["left"] = "open-palm-towards"

        # -------------------------------------------------------------
        #   PINCH-PRESSING -> pinch-holding
        # -------------------------------------------------------------
        elif current_state == "pinch-pressing":
            if gesture == "pinch":
                elapsed = current_time - self.hand_press_time["left"]
                if elapsed >= self.hold_threshold:
                    self.hand_state["left"] = "pinch-holding"
            else:
                # short pinch => short click
                elapsed = current_time - self.hand_press_time["left"]
                if elapsed < self.hold_threshold:
                    self.trigger_click_event()
                # end pinch => stop drawing
                self.canvas.stop_drawing()
                if palm_orientation == "palm_away":
                    self.hand_state["left"] = "open-palm-away"
                else:
                    self.hand_state["left"] = "open-palm-towards"

        # -------------------------------------------------------------
        #   PINCH-HOLDING -> remain or end
        # -------------------------------------------------------------
        elif current_state == "pinch-holding":
            if gesture == "pinch":
                # e.g., keep drawing
                pass
            else:
                # end pinch
                self.canvas.stop_drawing()
                if palm_orientation == "palm_away":
                    self.hand_state["left"] = "open-palm-away"
                else:
                    self.hand_state["left"] = "open-palm-towards"

        # -------------------------------------------------------------
        #   GRAB-AWAY: pressing -> holding
        # -------------------------------------------------------------
        elif current_state == "grab-away-pressing":
            if gesture == "grab_palm_away":
                elapsed = current_time - self.hand_press_time["left"]
                if elapsed >= self.hold_threshold:
                    self.hand_state["left"] = "grab-away-holding"
                    self.canvas.clear_gesture_screen()
            else:
                if palm_orientation == "palm_away":
                    self.hand_state["left"] = "open-palm-away"
                else:
                    self.hand_state["left"] = "open-palm-towards"

        elif current_state == "grab-away-holding":
            if gesture == "grab_palm_away":
                self.scroll_with_displacement(palm_y)
            else:
                if palm_orientation == "palm_away":
                    self.hand_state["left"] = "open-palm-away"
                else:
                    self.hand_state["left"] = "open-palm-towards"

        # -------------------------------------------------------------
        #   GRAB-TOWARDS: pressing -> holding
        # -------------------------------------------------------------
        elif current_state == "grab-towards-pressing":
            if gesture == "grab_palm_towards":
                elapsed = current_time - self.hand_press_time["left"]
                if elapsed >= self.hold_threshold:
                    self.hand_state["left"] = "grab-towards-holding"
                    self.canvas.clear_gesture_screen()
            else:
                if palm_orientation == "palm_away":
                    self.hand_state["left"] = "open-palm-away"
                else:
                    self.hand_state["left"] = "open-palm-towards"

        elif current_state == "grab-towards-holding":
            if gesture == "grab_palm_towards":
                self.scroll_with_displacement(palm_y)
            else:
                if palm_orientation == "palm_away":
                    self.hand_state["left"] = "open-palm-away"
                else:
                    self.hand_state["left"] = "open-palm-towards"

        # -------------------------------------------------------------
        #  Fallback
        # -------------------------------------------------------------
        else:
            if palm_orientation == "palm_away":
                self.hand_state["left"] = "open-palm-away"
            else:
                self.hand_state["left"] = "open-palm-towards"

        # -------------------------------------------------------------
        #  FINAL BLOCK: Thumbs-up / Thumbs-down detection
        # -------------------------------------------------------------
        if self.hand_state["left"]:
            if self.curr_hand is not None:
                thumb_gesture = self.check_thumb_gesture(self.curr_hand)
            else:
                thumb_gesture = None

            if thumb_gesture is not None:
                if self.hand_state["left"] != thumb_gesture:
                    self.hand_state["left"] = thumb_gesture
            else:
                if self.hand_state["left"] in ("thumbs_up", "thumbs_down"):
                    if palm_orientation == "palm_away":
                        self.hand_state["left"] = "open-palm-away"
                    else:
                        self.hand_state["left"] = "open-palm-towards"



    # --------------------
    # Setup mode
    # --------------------
    def setup_handler(self, event):
        """
        For each hand, expand max/min boundaries. Then save config.
        """
        for hand in event.hands:
            hp = hand.palm.position
            self.max_min_x[0] = max(self.max_min_x[0], hp.x)
            self.max_min_x[1] = min(self.max_min_x[1], hp.x)

            self.max_min_y[0] = max(self.max_min_y[0], hp.y)
            self.max_min_y[1] = min(self.max_min_y[1], hp.y)

            self.max_min_z[0] = max(self.max_min_z[0], hp.z)
            self.max_min_z[1] = min(self.max_min_z[1], hp.z)

        print("Current ranges: X =>", self.max_min_x, 
              " Y =>", self.max_min_y, 
              " Z =>", self.max_min_z)

        self.save_config()

    # --------------------
    # Utility: Move cursor
    # --------------------
    def move_cursor(self, x, y, scale_x, scale_y):
        """
        Skip mouse movement if enable_control == False.
        """
        if not self.enable_control:
            return
        
        if self.cap_width <= 0: 
            self.cap_width = 1
        if self.cap_height <= 0:
            self.cap_height = 1

        range_x = self.max_min_x[0] - self.max_min_x[1] or 1
        range_y = self.max_min_y[0] - self.max_min_y[1] or 1

        # Simple offset to shift the y range
        y_axis_offset = 0.1

        # Normalize to [0..1]
        norm_x = (x - self.max_min_x[1]) / range_x
        norm_y = ((y - self.max_min_y[1]) / range_y) - y_axis_offset

        screen_x = norm_x * self.screen_width  * scale_x
        screen_y = (1 - norm_y) * self.screen_height * scale_y

        # Clamp
        screen_x = max(0, min(screen_x, self.screen_width-1))
        screen_y = max(0, min(screen_y, self.screen_height-1))

        # Move OS cursor
        ctypes.windll.user32.SetCursorPos(int(screen_x), int(screen_y))

    # --------------------
    # Utility: Scrolling
    # --------------------
    def scroll_with_displacement(self, current_y):
        """
        Only scroll if OS control is enabled.
        """
        if not self.enable_control:
            return

        delta = current_y - self.last_scroll_y
        scroll_amount = int(-1 * delta * self.scroll_sensitivity * 120)
        if scroll_amount:
            MOUSEEVENTF_WHEEL = 0x0800
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, scroll_amount, 0)
        self.last_scroll_y = current_y

    # --------------------
    # Utility: Mouse Click
    # --------------------
    def trigger_click_event(self):
        """
        If we are not controlling the OS, skip the actual click.
        """
        if not self.enable_control:
            return
        ctypes.windll.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(self.MOUSEEVENTF_LEFTUP,   0, 0, 0, 0)

    def press_down(self):
        if not self.enable_control:
            return
        ctypes.windll.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def press_up(self):
        if not self.enable_control:
            return
        ctypes.windll.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
