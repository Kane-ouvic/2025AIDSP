import mediapipe as mp
print("✅ mediapipe imported")

mp_hands = mp.solutions.hands
print("✅ mp.solutions.hands loaded")

hands = mp_hands.Hands()
print("✅ Hands() instance created")