# 🐱🐶 Image Classification: Dogs vs. Cats Using Deep Learning  

## 📌 Project Overview  
This project builds a **Convolutional Neural Network (CNN)** model using **TensorFlow/Keras** to classify images of cats and dogs. The dataset is sourced from `PetImages`, and the model is trained using **data augmentation** and **transfer learning** to improve accuracy.  
![dog](https://github.com/user-attachments/assets/7d7702f2-a11c-4f59-9342-35e2e45bcb0a)
![cat](https://github.com/user-attachments/assets/3fb230e5-2135-4530-9402-6827b5357057)

---

## 📂 Dataset  
- The dataset consists of images of **cats (label = 0)** and **dogs (label = 1)**.  
- Data is split into **training (80%)** and **testing (20%)** using `train_test_split`.  
- Data augmentation techniques such as **rotation, zoom, and flipping** are used to improve model generalization.  

---

## 🏗️ Model Architecture  
The CNN architecture includes:  
✔ **Conv2D + MaxPooling layers** for feature extraction  
✔ **Flatten layer** to convert feature maps into a dense representation  
✔ **Fully connected layers with ReLU activation**  
✔ **Sigmoid activation for binary classification**  

```python  
from keras import Sequential  
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  

model = Sequential([  
    Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),  
    MaxPooling2D((2,2)),  
    Conv2D(32, (3,3), activation='relu'),  
    MaxPooling2D((2,2)),  
    Conv2D(64, (3,3), activation='relu'),  
    MaxPooling2D((2,2)),  
    Flatten(),  
    Dense(512, activation='relu'),  
    Dense(1, activation='sigmoid')  # Output layer  
])  
```

---

## 🏋️ Training the Model  
The model is compiled with:  
- **Binary Crossentropy Loss** (for two classes)  
- **Adam Optimizer** (efficient weight updates)  
- **Accuracy as the evaluation metric**  

```python  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)  
```

---

## 📊 Results and Evaluation  
- **Loss & Accuracy Graphs** are plotted after training.  
- The trained model is evaluated on unseen test data.  
- **Predictions are made on real images.**  

```python  
import matplotlib.pyplot as plt  

# Plot Accuracy  
plt.plot(history.history['accuracy'], label='Training Accuracy')  
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  
plt.title('Accuracy Graph')  
plt.legend()  
plt.show()

# Plot Loss  
plt.plot(history.history['loss'], label='Training Loss')  
plt.plot(history.history['val_loss'], label='Validation Loss')  
plt.title('Loss Graph')  
plt.legend()  
plt.show()  
```

---

## 🛠️ How to Run This Project  

### **1️⃣ Clone this repository:**  
```bash  
git clone https://github.com/your-username/image-classification-dogs-cats.git  
```

### **2️⃣ Install dependencies:**  
```bash  
pip install -r requirements.txt  
```

### **3️⃣ Run the training script:**  
```bash  
python train.py  
```

### **4️⃣ Test with an image:**  
```python  
image_path = "test.jpg"  
img = load_img(image_path, target_size=(128, 128))  
img = np.array(img) / 255.0  # Normalize  
img = img.reshape(1, 128, 128, 3)  
pred = model.predict(img)  
label = "Dog 🐶" if pred[0] > 0.5 else "Cat 🐱"  
print(label)  
```

![output dog itesting](https://github.com/user-attachments/assets/76a48401-cd13-4a8e-b9ab-3534aacfb2da)
![cat testing output](https://github.com/user-attachments/assets/603a5da6-e40d-4602-833b-ae419a63483a)

---

## 📁 File Structure  
```
image-classification-dogs-cats/
│── dataset/                     # Folder containing images
│   ├── PetImages/
│   │   ├── Cat/                 # Cat images
│   │   ├── Dog/                 # Dog images
│── notebooks/                    # Jupyter Notebooks for training & testing
│── models/                       # Trained model files
│── scripts/                      # Python scripts for preprocessing & training
│── train.py                       # Model training script
│── test.py                        # Model inference script
│── requirements.txt               # List of dependencies
│── README.md                      # Project documentation
│── LICENSE                        # Open-source license
```

---

## 🎯 Topics & Tags  
- `deep-learning`  
- `tensorflow`  
- `image-classification`  
- `convolutional-neural-networks`  
- `computer-vision`  
- `machine-learning`  
- `cats-vs-dogs`  

---

## 📜 License  
This project is open-source under the **MIT License**.  

```markdown  
MIT License  

Copyright (c) 2024 [Your Name]  

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software to use, copy, modify, merge, publish, and distribute the software  
without restriction, subject to the following conditions:  

- The above copyright notice and this permission notice shall be included in all copies.  
- The software is provided "AS IS", without warranty of any kind.  
```  

---

### 🎯 **Now Just Push This to GitHub!** 🚀  
This **README + file structure + license** will make your project **professional & well-documented**.  

Let me know if you need changes! 😊

