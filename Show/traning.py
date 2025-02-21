import os
import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from tqdm import tqdm

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model FaceNet
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, device=device)

# Định nghĩa đường dẫn dataset và model
DATASET_DIR = r'E:/IOT/project_Binh/project_Binh/dataset'
MODEL_PATH = 'E:/IOT/project_Binh/project_Binh/Show/models/face_recognition_model1.pth'
EMBEDDINGS_PATH = 'E:/IOT/project_Binh/project_Binh/Show/models/embeddings_data.pth'

# Kiểm tra xem embeddings đã có dữ liệu cũ không
if os.path.exists(EMBEDDINGS_PATH):
    old_data = torch.load(EMBEDDINGS_PATH)
    old_embeddings = old_data['embeddings']
    old_labels = old_data['labels']
else:
    old_embeddings = np.array([])
    old_labels = np.array([])

# Chuẩn bị transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

new_embeddings = []
new_labels = []

for label in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, label)
    if os.path.isdir(person_dir):
        embeddings_list = []
        for image_name in tqdm(os.listdir(person_dir), desc=f'Processing {label}'):
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # Phát hiện khuôn mặt
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                for box in boxes:
                    face_crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    face_tensor = transform(face_crop).unsqueeze(0).to(device)
                    embedding = model(face_tensor).detach().cpu().numpy().flatten()
                    embeddings_list.append(embedding)
        
        if embeddings_list:
            avg_embedding = np.mean(embeddings_list, axis=0)
            new_embeddings.append(avg_embedding)
            new_labels.append(label)

# Kết hợp dữ liệu cũ và mới
if old_embeddings.size > 0:
    combined_embeddings = np.concatenate((old_embeddings, np.array(new_embeddings)), axis=0)
    combined_labels = np.concatenate((old_labels, np.array(new_labels)), axis=0)
else:
    combined_embeddings = np.array(new_embeddings)
    combined_labels = np.array(new_labels)

# Lưu embeddings và labels mới
face_data = {
    'embeddings': combined_embeddings,
    'labels': combined_labels
}
torch.save(face_data, EMBEDDINGS_PATH)

torch.save(model.state_dict(), MODEL_PATH)

print("Training hoàn tất! Dữ liệu embeddings đã được lưu vào", EMBEDDINGS_PATH)
print("Mô hình đã được lưu vào", MODEL_PATH)
