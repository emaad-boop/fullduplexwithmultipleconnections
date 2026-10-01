import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from ultralytics import YOLO

print("Loading YOLOv8 and Haar Cascade engines...")
yolo_model = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


async def stream_handler(websocket):
    print("A client has successfully connected over the full-duplex tunnel!")
    last_face_snippet = None

    try:
        async for message in websocket:
            data = json.loads(message)
            img_data = data['image']

            if isinstance(img_data, list):
                img_data = img_data[-1]
            if isinstance(img_data, str) and ',' in img_data:
                img_data = img_data.split(',')[-1]

            try:
                img_bytes = base64.b64decode(img_data)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None or frame.size == 0:
                continue

            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_frame = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            face_as_text = None
            face_detected_this_frame = False

            try:
                # Run person tracking boundaries
                yolo_results = yolo_model(output_frame, classes=[0], conf=0.3, verbose=False)

                for result in yolo_results:
                    boxes = result.boxes
                    for box in boxes:
                        # Defensive Extraction: Flatten and safely convert coordinates to flat list integers
                        coords = box.xyxy.cpu().numpy().flatten().astype(int).tolist()
                        if len(coords) < 4:
                            continue

                        hx1, hy1, hx2, hy2 = coords[0], coords[1], coords[2], coords[3]

                        # Clip boundaries cleanly to image size array limits
                        h_max, w_max = frame.shape[0], frame.shape[1]
                        hy1, hy2 = max(0, hy1), min(h_max, hy2)
                        hx1, hx2 = max(0, hx1), min(w_max, hx2)

                        human_crop = gray_img[hy1:hy2, hx1:hx2]
                        if human_crop.size == 0:
                            continue

                        # Scan for face features inside the human bounding frame
                        faces = face_cascade.detectMultiScale(human_crop, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                        for (fx, fy, fw, fh) in faces:
                            face_detected_this_frame = True

                            # Convert coordinates back into the global tracking scene
                            face_x1 = hx1 + fx
                            face_y1 = hy1 + fy
                            face_x2 = face_x1 + fw
                            face_y2 = face_y1 + fh

                            face_y1, face_y2 = max(0, face_y1), min(h_max, face_y2)
                            face_x1, face_x2 = max(0, face_x1), min(w_max, face_x2)

                            face_snippet = output_frame[face_y1:face_y2, face_x1:face_x2]
                            if face_snippet.size > 0:
                                _, face_buffer = cv2.imencode('.jpg', face_snippet)
                                face_as_text = base64.b64encode(face_buffer).decode('utf-8')
                                last_face_snippet = face_as_text

                            cv2.rectangle(output_frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)
                            cv2.putText(output_frame, 'Human Face', (face_x1, face_y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as inner_e:
                # If low lighting confuses coordinate dimensions, swallow error and keep streaming alive
                pass

            _, buffer = cv2.imencode('.jpg', output_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            payload = {
                'image': jpg_as_text,
                'face': face_as_text if face_as_text else last_face_snippet,
                'face_detected': face_detected_this_frame
            }
            await websocket.send(json.dumps(payload))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected from pipe gracefully.")
    except Exception as e:
        print(f"Data stream handling anomaly caught: {e}")


async def main():
    print("Starting processing server on port 8766...")
    async with websockets.serve(stream_handler, "0.0.0.0", 8766, max_size=10_000_000):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer gracefully shut down.")
