import cv2
import base64
import time

print("Testing camera access...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("FAIL: Camera 0 with CAP_DSHOW failed, trying without...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("FAIL: Cannot open camera at all. Try index 1 or 2.")
    exit()

print("OK: Camera opened")

ret, frame = cap.read()
if not ret or frame is None:
    print("FAIL: Could not read frame")
    cap.release()
    exit()

print(f"OK: Frame read — shape={frame.shape}, dtype={frame.dtype}")

ok, jpeg = cv2.imencode(".jpg", frame)
if not ok:
    print("FAIL: imencode failed")
    cap.release()
    exit()

b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
print(f"OK: JPEG encoded — {len(jpeg.tobytes())} bytes, base64 length={len(b64)}")

# Save a test frame so user can visually verify
cv2.imwrite("test_frame.jpg", frame)
print("OK: Saved test_frame.jpg — open it to verify camera is working")

cap.release()
print("\nAll checks passed. Camera is working correctly for streaming.")