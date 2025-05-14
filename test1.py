import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

def calc_angle(a, b, c):
    ab = [b[i] - a[i] for i in range(3)]
    cb = [b[i] - c[i] for i in range(3)]
    dot = sum(ab[i] * cb[i] for i in range(3))
    ab_len = math.sqrt(sum(x**2 for x in ab))
    cb_len = math.sqrt(sum(x**2 for x in cb))
    angle = math.acos(dot / (ab_len * cb_len + 1e-6))
    return math.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            if hand_info.classification[0].label == 'Left':
                lm = hand_landmarks.landmark
                angle = calc_angle(
                    [lm[5].x, lm[5].y, lm[5].z],
                    [lm[6].x, lm[6].y, lm[6].z],
                    [lm[8].x, lm[8].y, lm[8].z]
                )
                cv2.putText(image, f'Index Angle: {angle:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
