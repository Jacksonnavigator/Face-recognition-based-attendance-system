import os
import cv2
import numpy as np
import pandas as pd
import joblib
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Set constants and initialize variables
nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initialize face detector
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

# Ensure necessary directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')

# Helper functions
def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return face_points

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    for user in userlist:
        name, roll = user.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, len(userlist)

def deletefolder(duser):
    pics = os.listdir(duser)
    for pic in pics:
        os.remove(duser + '/' + pic)
    os.rmdir(duser)

# Streamlit App
st.title('Face Recognition Attendance System')

menu = ['Home', 'Add User', 'List Users', 'Delete User', 'Start Attendance']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
    names, rolls, times, l = extract_attendance()
    st.header('Attendance Records')
    df = pd.DataFrame({'Name': names, 'Roll': rolls, 'Time': times})
    st.table(df)
    st.text(f'Total Registered Users: {totalreg()}')

elif choice == 'Add User':
    st.header('Add a New User')
    newusername = st.text_input('Enter Name:')
    newuserid = st.text_input('Enter Roll Number:')
    
    if st.button('Add User'):
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        os.makedirs(userimagefolder, exist_ok=True)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while i < nimgs:
            ret, frame = cap.read()
            if ret:
                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    if j % 5 == 0:
                        name = newusername + '_' + str(i) + '.jpg'
                        cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                        i += 1
                    j += 1
                st.image(frame, channels="BGR")
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        train_model()
        st.success('User added and model trained!')

elif choice == 'List Users':
    userlist, names, rolls, l = getallusers()
    st.header('List of Users')
    for name, roll in zip(names, rolls):
        st.text(f'{name} ({roll})')
    st.text(f'Total Registered Users: {totalreg()}')

elif choice == 'Delete User':
    userlist, names, rolls, l = getallusers()
    user_to_delete = st.selectbox('Select User to Delete', userlist)
    
    if st.button('Delete User'):
        deletefolder('static/faces/' + user_to_delete)
        if not os.listdir('static/faces'):
            os.remove('static/face_recognition_model.pkl')
        train_model()
        st.success(f'User {user_to_delete} deleted and model retrained!')

elif choice == 'Start Attendance':
    st.header('Start Attendance')
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        st.warning('There is no trained model in the static folder. Please add a new face to continue.')
    else:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
                    face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                    identified_person = identify_face(face.reshape(1, -1))[0]
                    add_attendance(identified_person)
                    cv2.putText(frame, f'{identified_person}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                st.image(frame, channels="BGR")
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        st.success('Attendance updated!')

