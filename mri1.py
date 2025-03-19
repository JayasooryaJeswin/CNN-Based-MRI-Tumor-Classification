import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from skimage import io, transform

images = np.load('C:/JAYASOORYA/RESEARCH SOURCE/mri_images.npy')
labels = np.load('C:/JAYASOORYA/RESEARCH SOURCE/mri_labels.npy')

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_images, train_labels,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

def preprocess_image(image_path):
    image = io.imread(image_path, as_gray=True)  
    image = transform.resize(image, (128, 128))  
    image = np.expand_dims(image, axis=-1)       
    image = np.expand_dims(image, axis=0)        
    return image


new_image_path = "C:/JAYASOORYA/RESEARCH SOURCE/yes/Y85.JPG"  

new_image = preprocess_image(new_image_path)

prediction = model.predict(new_image)
prediction_value = prediction[0][0] 

diagnosis = "Tumor" if prediction_value < 0.5 else "No Tumor" 
confidence = prediction_value if prediction_value < 0.5 else 1 - prediction_value

print(f"Diagnosis: {diagnosis} (Confidence: {confidence * 100:.2f}%)")