from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
import sys
import face_recognition
import numpy as np

from detector import detect_face, detect_face_location
from embedder import load_embedder, get_face_embedding
from emb_store import load_embeddings, save_embeddings, find_match,create_empty_embeddings_json

import cv2

class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.embedder = load_embedder('models/embedding.keras')
        self.embeddings_db = load_embeddings('embeddings.json')
        self.current_embedding = None
        self.current_image = None
        self.current_name = None
        self.current_face_location = None

    def init_ui(self):
        self.setWindowTitle("Face Recognition App")
        layout = QVBoxLayout()

        self.image_label = QLabel("No image loaded")
        self.image_label.setFixedSize(400, 400)
        layout.addWidget(self.image_label)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        btn_layout = QHBoxLayout()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        btn_layout.addWidget(self.btn_load)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name to add")
        btn_layout.addWidget(self.name_input)

        self.btn_add = QPushButton("Add to DB")
        self.btn_add.clicked.connect(self.add_to_db)
        btn_layout.addWidget(self.btn_add)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.jpg *.jpeg *.png)")
        if file_path:
            # Загружаем изображение через OpenCV
            bgr_img = cv2.imread(file_path)
            if bgr_img is None:
                self.result_label.setText("Failed to load image")
                return
                
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
            face = detect_face(rgb_img)
            if face is None:
                self.result_label.setText("No face detected")
                return

            face_location = detect_face_location(rgb_img)
            if face_location is None:
                self.result_label.setText("Face detected but location not found")
                return

            embedding = get_face_embedding(self.embedder, face)
            self.current_embedding = embedding
            self.current_image = rgb_img
            self.current_face_location = face_location

            name = find_match(embedding, self.embeddings_db)
            self.current_name = name

            self.result_label.setText(f"Recognized: {name}")
            self.display_image_with_box(rgb_img, face_location, name)

    def add_to_db(self):
        name = self.name_input.text().strip()
        if not name or self.current_embedding is None:
            return
        self.embeddings_db[name] = self.current_embedding.tolist()
        save_embeddings('embeddings.json', self.embeddings_db)
        self.result_label.setText(f"Added {name} to DB")

    def display_image_with_box(self, img, face_location, name):
        top, right, bottom, left = face_location
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        painter = QPainter(pixmap)
        pen = QPen(QColor("green"))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRect(left, top, right - left, bottom - top)

        painter.setPen(QColor("red"))
        painter.drawText(left, top - 10, name)

        painter.end()
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=1))

if __name__ == '__main__':
    create_empty_embeddings_json()
    app = QApplication(sys.argv)
    win = FaceApp()
    win.show()
    sys.exit(app.exec())
