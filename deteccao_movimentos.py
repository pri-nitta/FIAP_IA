import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm
from deepface import deepface
import face_recognition

def load_images_from_folder(folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                face_encoding = face_encodings[0]
                name = os.path.splitext(filename)[0][:-1]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names

def detect_pose_and_emotions(video_path, output_path, known_face_encodings, known_face_names):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    pose = mp_pose.Pose()
    hands = mp_hands.Hands()
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    arm_up = False
    arm_movements_count = 0
    eyebrow_raised_count = 0
    eyebrow_raised = False
    looking_down_count = 0
    looking_down = False

    emotion_counts = {
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0
    }

    def is_arm_up(landmarks):
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        left_arm_up = left_elbow.y < left_eye.y
        right_arm_up = right_elbow.y < right_eye.y
        return left_arm_up and right_arm_up

    def is_raising_eyebrow(face_landmarks):
        left_eyebrow = face_landmarks.landmark[70]
        right_eyebrow = face_landmarks.landmark[300]
        left_eye = face_landmarks.landmark[159]
        right_eye = face_landmarks.landmark[386]

        if left_eyebrow.y < left_eye.y - 0.05 or right_eyebrow.y < right_eye.y - 0.05:
            return True
        return False

    def is_looking_down(face_landmarks):
        nose = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]

        if nose.y < chin.y - 0.1:
            return True
        return False

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)
        face_mesh_results = face_mesh.process(rgb_frame)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if is_arm_up(pose_results.pose_landmarks.landmark):
                if not arm_up:
                    arm_up = True
                    arm_movements_count += 1
            else:
                arm_up = False

            cv2.putText(frame, f'Movimento dos bracos: {arm_movements_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                if is_raising_eyebrow(face_landmarks):
                    if not eyebrow_raised:
                        eyebrow_raised = True
                        eyebrow_raised_count += 1
                else:
                    eyebrow_raised = False

                cv2.putText(frame, f'Sobrancelhas erguidas: {eyebrow_raised_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                if is_looking_down(face_landmarks):
                    if not looking_down:
                        looking_down = True
                        looking_down_count += 1
                else:
                    looking_down = False

                cv2.putText(frame, f'Olhando para baixo: {looking_down_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconhecido"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        for face in result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            dominant_emotion = face['dominant_emotion']
            emotion_counts[dominant_emotion] += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if x <= left <= x + w and y <= top <= y + h:
                    cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Contagem de emoções:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count}")

# Caminho para a pasta de imagens com rostos conhecidos
image_folder = 'images'

# Carregar imagens e codificações
known_face_encodings, known_face_names = load_images_from_folder(image_folder)

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')
output_video_path = os.path.join(script_dir, 'output_video_pose_emotions.mp4')

# Chamar a função para detectar poses, emoções e reconhecer faces no vídeo, salvando o vídeo processado
detect_pose_and_emotions(input_video_path, output_video_path, known_face_encodings, known_face_names)