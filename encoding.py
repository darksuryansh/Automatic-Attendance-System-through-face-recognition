import cv2
import face_recognition
import pickle
import os
from tqdm import tqdm  # for progress bar

# Initialize variables to store encodings and names
known_face_encodings = []
known_face_names = []

# Path to the directory containing all person folders
base_dir = 'data/images'

# Loop through each person in the dataset
for person_name in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person_name)
    
    # Skip if it's not a directory
    if not os.path.isdir(person_dir):
        continue
    
    print(f"Processing {person_name}'s images...")
    
    # Process each image of the person
    for image_file in tqdm(os.listdir(person_dir)):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(person_dir, image_file)
            
            # Load the image
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings for the image
            face_encodings = face_recognition.face_encodings(image)
            
            # If at least one face is found, use the first one
            if len(face_encodings) > 0:
                encoding = face_encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)

# Save the face encodings and names to a pickle file
encodings_data = {
    "encodings": known_face_encodings,
    "names": known_face_names
}

with open('data/encodings.pkl', 'wb') as f:
    pickle.dump(encodings_data, f)

print(f"\nEncoding process completed!")
print(f"Total faces encoded: {len(known_face_encodings)}")
print(f"Saved encodings to data/face_encodings.pkl")