def recognize_and_display_table():
    cap = cv2.VideoCapture(0)
    frame_count = 0  # Add frame counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 5th frame
        if frame_count % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Matching logic here...

        # Display every frame
        cv2.imshow("Face Recognition", frame)

        frame_count += 1  # Increment frame counter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
