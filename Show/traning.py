import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model FaceNet
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, device=device)

# Định nghĩa đường dẫn dataset và model
DATASET_DIR = "dataset"
MODEL_PATH = "Show/models/face_recognition_model1.pth"
EMBEDDINGS_PATH = "Show/models/embeddings_data.pth"

# Chuẩn bị transform ảnh
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def train_new_student(folder_name):
    person_dir = os.path.join(DATASET_DIR, folder_name)

    if not os.path.isdir(person_dir):
        return False, "Thư mục sinh viên không tồn tại"

    # Kiểm tra số lượng ảnh
    if len(os.listdir(person_dir)) < 3:
        return False, "Cần ít nhất 5 ảnh của sinh viên để training"

    # Load embeddings cũ
    if os.path.exists(EMBEDDINGS_PATH):
        old_data = torch.load(EMBEDDINGS_PATH)
        old_embeddings = old_data["embeddings"]
        old_labels = old_data["labels"]
    else:
        old_embeddings = np.array([])
        old_labels = np.array([])

    embeddings_list = []
    for image_name in tqdm(os.listdir(person_dir), desc=f"Processing {folder_name}"):
        image_path = os.path.join(person_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        # Phát hiện khuôn mặt
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            for box in boxes:
                face_crop = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                face_tensor = transform(face_crop).unsqueeze(0).to(device)
                embedding = model(face_tensor).detach().cpu().numpy().flatten()
                embeddings_list.append(embedding)

    if embeddings_list:
        avg_embedding = np.mean(embeddings_list, axis=0)

        # Thêm vào dữ liệu hiện tại
        if old_embeddings.size > 0:
            old_embeddings = np.concatenate(
                (old_embeddings, np.array([avg_embedding])), axis=0
            )
            old_labels = np.concatenate((old_labels, np.array([folder_name])), axis=0)
        else:
            old_embeddings = np.array([avg_embedding])
            old_labels = np.array([folder_name])

        # Lưu lại embeddings
        face_data = {"embeddings": old_embeddings, "labels": old_labels}
        torch.save(face_data, EMBEDDINGS_PATH)

        # Lưu trạng thái mô hình
        torch.save(model.state_dict(), MODEL_PATH)

        return True, f"Thêm embeddings cho {folder_name} thành công!"

    return False, "Không tìm thấy khuôn mặt nào trong ảnh"

