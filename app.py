import streamlit as st
import cv2
import mediapipe as mp
import sqlite3
import numpy as np
from PIL import Image
import face_recognition
import io
import time
from streamlit_option_menu import option_menu

# Database path
DB_PATH = 'restaurant_bookings.db'

# Function to set up the database
def setup_database():
    """Sets up the database with a table for storing user information."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                phone TEXT,
                pax INTEGER,
                table_number INTEGER,
                face_encoding BLOB,
                image BLOB
            )
        """)
        conn.commit()

# Function to update the database schema
def update_database_schema():
    """Add columns to the bookings table if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE bookings ADD COLUMN face_encoding BLOB")
            cursor.execute("ALTER TABLE bookings ADD COLUMN image BLOB")
            conn.commit()
        except sqlite3.OperationalError:
            # Columns already exist
            pass

# Call both functions to ensure the database is properly set up
setup_database()
update_database_schema()

# Setup Mediapipe for face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to get the list of assigned tables
def get_assigned_tables():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT table_number FROM bookings")
        rows = cursor.fetchall()
        assigned_tables = [row[0] for row in rows]
    return assigned_tables

# Function to allocate a table based on pax count and availability
def allocate_table(pax):
    assigned_tables = get_assigned_tables()
    if pax <= 2 and 1 not in assigned_tables:
        return 1
    elif pax <= 4 and 2 not in assigned_tables:
        return 2
    elif pax > 4 and 3 not in assigned_tables:
        return 3
    else:
        return None  # No available table

# Function to encode face and store it
def encode_face(image_bytes):
    """Encodes the face from an image."""
    image_np = np.array(Image.open(io.BytesIO(image_bytes)))
    face_locations = face_recognition.face_locations(image_np)
    if face_locations:
        face_encoding = face_recognition.face_encodings(image_np, face_locations)[0]
        return face_encoding
    return None

# Function to register a user and save data in the database
def register_user(name, phone, pax, image_bytes):
    table_number = allocate_table(pax)
    if table_number is None:
        st.warning("No available table for the specified pax. Please try again later.")
        return None

    face_encoding = encode_face(image_bytes)
    if face_encoding is not None:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bookings (name, phone, pax, table_number, face_encoding, image) VALUES (?, ?, ?, ?, ?, ?)",
                (name, phone, pax, table_number, face_encoding.tobytes(), image_bytes)
            )
            conn.commit()
        return table_number
    else:
        st.warning("No face detected in the image. Please try again.")
        return None

# Function to delete a guest from the database
def delete_guest(guest_id):
    """Deletes a guest from the database by their ID."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bookings WHERE id = ?", (guest_id,))
        conn.commit()

# Function to load encodings from the database
def load_encodings_from_db():
    """Loads face encodings and associated data from the database."""
    encodings = []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, phone, pax, table_number, face_encoding FROM bookings")
        rows = cursor.fetchall()
        for row in rows:
            name, phone, pax, table_number, face_encoding = row
            if face_encoding:  # Only load valid encodings
                encodings.append({
                    "name": name,
                    "phone": phone,
                    "pax": pax,
                    "table_number": table_number,
                    "face_encoding": np.frombuffer(face_encoding, dtype=np.float64)
                })
    return encodings

# Streamlit UI with multiple tabs
st.sidebar.title("Welcome to F&B Booking System")

#menu = st.sidebar.radio("Select menu", ["Register Guest", "View Guests", "Search Guest", "Camera Detection"])
with st.sidebar:
    menu = option_menu(
        menu_title = "menu",
        options = ["Register Guest", "View Guests", "Search Guest", "Camera Detection"],
        icons = ["clipboard-data", "graph-up-arrow", "calculator-fill", "cart3"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "vertical"
    )
if menu == "Register Guest":
    st.title("Guest Registration")
    
    # Registration form
    name = st.text_input("Enter Full Name")
    phone = st.text_input("Enter Phone Number")
    pax = st.number_input("Number of People (Pax)", min_value=1, max_value=10)
    
    # Camera capture for taking a picture
    st.write("Take a picture of yourself for registration:")
    camera_image = st.camera_input("Take a Picture")
    
    # Terms and conditions message
    st.write("The information you upload will be stored only for processing purposes. "
             "The information you upload may be shared. Do not upload important information. "
             "We cannot be held responsible.")
    
    # Terms and conditions agreement
    agree = st.checkbox("I understand and agree")

    if st.button("Register"):
        if name and phone and pax and camera_image and agree:
            image_bytes = camera_image.getvalue()
            table_number = register_user(name, phone, pax, image_bytes)
            if table_number:
                st.success(f"Guest registered! Assigned Table: {table_number}")
        elif not agree:
            st.warning("You must agree to the terms and conditions to register.")
        else:
            st.warning("Please fill in all fields and take a photo.")

elif menu == "View Guests":
    st.title("View Guests")

    # Fetch guest data from the database
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, phone, pax, table_number, image FROM bookings")
        rows = cursor.fetchall()

    if rows:
        st.write("Loading list of guests...")
        st.success("Loading successful: Load successful")

        for row in rows:
            guest_id, name, phone, pax, table_number, image_data = row
            st.write(f"**Name**: {name}")
            st.write(f"**Phone**: {phone}")
            st.write(f"**Pax**: {pax}")
            st.write(f"**Table**: {table_number}")

            # Convert image data back to an image for display
            if image_data:
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption=f"Registration Time: [Generated Timestamp]", use_column_width=True)

            # Option to delete the guest
            if st.button("Delete", key=f"delete_{guest_id}"):
                delete_guest(guest_id)
                st.success(f"Deleted guest: {name}")
            st.write("---")
    else:
        st.warning("No guests registered yet.")

elif menu == "Search Guest":
    st.title("Search Guest by Name")
    
    search_name = st.text_input("Enter the name to search for")
    
    if st.button("Search"):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, phone, pax, table_number FROM bookings WHERE name LIKE ?", (f"%{search_name}%",))
            results = cursor.fetchall()
            
            if results:
                for result in results:
                    name, phone, pax, table_number = result
                    st.write(f"**Name**: {name}")
                    st.write(f"**Phone**: {phone}")
                    st.write(f"**Pax**: {pax}")
                    st.write(f"**Table**: {table_number}")
                    st.write("---")
            else:
                st.write("No matching guests found.")

elif menu == "Camera Detection":
    st.title("Real-Time Face Recognition")
    
    encodings_db = load_encodings_from_db()
    if not encodings_db:
        st.write("No registered guests to recognize.")
    else:
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_webcam = False

        # Single "Stop Webcam" button outside the loop to avoid multiple instances
        if st.button("Stop Webcam"):
            stop_webcam = True

        while cap.isOpened() and not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB format for face recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Process each detected face
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(
                    [enc["face_encoding"] for enc in encodings_db], face_encoding
                )
                name = "Unknown"
                pax = "N/A"
                table_number = "N/A"

                if True in matches:
                    match_index = matches.index(True)
                    matched_guest = encodings_db[match_index]
                    name = matched_guest["name"]
                    pax = matched_guest["pax"]
                    table_number = matched_guest["table_number"]

                # Get face location coordinates
                top, right, bottom, left = face_location
                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Display guest information in the top-left corner
                cv2.putText(frame, f"Name: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Pax: {pax}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Table: {table_number}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Update the Streamlit image display
            stframe.image(frame, channels="BGR")

            # Add a small sleep delay to improve performance
            time.sleep(0.03)

        cap.release()
        cv2.destroyAllWindows()
