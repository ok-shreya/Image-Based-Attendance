from flask import Flask, request, render_template
import os
import pandas as pd
from datetime import datetime
import pickle
import cv2
import numpy as np
import face_recognition
from mtcnn import MTCNN
import tensorflow as tf

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = Flask(_name_)

app.config['UPLOAD_FOLDER'] = 'images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_image_from_path(image_path):
    img = cv2.imread(image_path)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit_date():
    if request.method == 'POST':
        manual_date = request.form['date']
        periods = request.form['period']
        stream = request.form['stream']
        course = request.form['course']
        semester = request.form['semester']
        paper = request.form['paper']

        try:
            datetime.strptime(manual_date, '%Y-%m-%d')
        except ValueError:
            return "Date format is incorrect. It should be 'YYYY-MM-DD'."

        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = 'uploaded_image.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

        detector = MTCNN()

        student_dict = {
            925: "SK KAIF RAHAMAN",
            1053: "MITODRU MRIDHA",
            # ... (other student entries)
        }

        encode_file_path = "EncodeFile.p"
        with open(encode_file_path, 'rb') as file:
            encodeListKnown_Ids = pickle.load(file)
        encodeListKnown, studentIds = encodeListKnown_Ids

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = load_image_from_path(image_path)

        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        detections = detector.detect_faces(img_small_rgb)
        face_locations = [(d['box'][1], d['box'][0] + d['box'][2], d['box'][1] + d['box'][3], d['box'][0]) for d in detections]
        face_encodings = [face_recognition.face_encodings(img_small_rgb, [face_location])[0] for face_location in face_locations]

        total_faces_detected = len(face_locations)
        known_faces_detected = 0

        recognized_student_names = []

        for face_location, face_encoding in zip(face_locations, face_encodings):
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            best_match_id = None
            best_confidence = 0

            for encodeKnown, studentId in zip(encodeListKnown, studentIds):
                distance = face_recognition.face_distance([encodeKnown], face_encoding)
                confidence = (1 - distance[0]) * 100
                if confidence > 47 and confidence > best_confidence:
                    best_confidence = confidence
                    best_match_id = studentId

            if best_match_id is not None:
                cv2.putText(img, f"{best_match_id} - {student_dict.get(best_match_id, '')}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                known_faces_detected += 1
                recognized_student_names.append(student_dict.get(best_match_id, best_match_id))
            else:
                cv2.putText(img, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        height, width = img.shape[:2]
        new_width = 1080
        new_height = int((new_width / width) * height)
        resized_img = cv2.resize(img, (new_width, new_height))

        if paper == 'CC13':
            excel_path = "CC13.xlsx"
        elif paper == 'CC14':
            excel_path = "CC14.xlsx"
        elif paper == 'DSE-A':
            excel_path = "DSE-A.xlsx"
        elif paper == 'DSE-B':
            excel_path = "DSE-B.xlsx"
        
        try:
            df = pd.read_excel(excel_path)
        except PermissionError as e:
            print(f"Permission error: {e}")
            exit()

        current_date = manual_date

        if current_date not in df.columns:
            df[current_date] = 0

        for name in recognized_student_names:
            df.loc[df['Student Name'] == name.strip(), current_date] = periods

        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
        except PermissionError as e:
            print(f"Permission error: {e}")
            exit()

        if os.name == 'nt':
            os.startfile(excel_path)
        elif os.name == 'posix':
            os.system(f'open "{excel_path}"' if sys.platform == 'darwin' else f'xdg-open "{excel_path}"')
        else:
            print(f'Cannot open file automatically on this OS: {os.name}')

        cv2.imshow("Face attendance", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return render_template('image_upload_page.html')

if _name_ == '_main_':
    app.run(debug=True)