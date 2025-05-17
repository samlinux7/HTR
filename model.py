import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from keras.saving import register_keras_serializable
import string
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2
import random
import os
import matplotlib.pyplot as plt


# CONFIGURATION
IMG_WIDTH = 128
IMG_HEIGHT = 32
BATCH_SIZE = 32
EPOCHS = 30
MAX_LABEL_LENGTH = 40

# Paths to .npy files
IMAGES_NPY = "images.npy"
LABELS_NPY = "labels.npy"

CHARS = string.ascii_letters + string.digits + "!?.:,;'-" + " "
CHARSET = set(CHARS)


# Character Mappings
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(CHARS), oov_token="[UNK]")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True)



# CRNN Model (Prediction Model)
@register_keras_serializable()
class CRNN(Model):
    def __init__(self, **kwargs):
        super(CRNN, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.Activation('relu')
        self.reshape = layers.Reshape((IMG_WIDTH // 4, (IMG_HEIGHT // 4) * 256))
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.bilstm1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))
        self.dense = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)
        x = self.reshape(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.bilstm1(x)
        return self.dense(x)
    
    def get_config(self):
        config = super(CRNN, self).get_config()
        # Add any custom config values if needed in future
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


which_model = "60"

if which_model == '20':
    model = tf.keras.models.load_model('crnn_model.keras', custom_objects={"CRNN": CRNN})
elif which_model == '60':
    model = tf.keras.models.load_model('crnn_model_60.keras', custom_objects={"CRNN": CRNN})
else:
    print("Wrong model")
    SystemExit()


# Encode labels
def encode_label(label):
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    return char_to_num(label)


# Preprocess image
def preprocess_image(img_path, augment=False):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0

        # Lightweight augmentation for training
        if augment and random.random() < 0.5:
            # Random rotation (±5 degrees)
            angle = random.uniform(-5, 5)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=1.0)  # White background
            # Random brightness
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0.0, 1.0)

        img = np.expand_dims(img, axis=-1)
        return img
    except:
        return None



# Data Generator
class DataGenerator(Sequence):
    def __init__(self, data, batch_size, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(data))

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = np.zeros((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        labels = np.ones((self.batch_size, MAX_LABEL_LENGTH), dtype=np.int32) * -1
        label_lengths = np.zeros((self.batch_size,), dtype=np.int32)

        for i, (img_path, text) in enumerate(batch_data):
            img = preprocess_image(img_path, augment=self.augment)
            if img is None:
                continue
            images[i] = img
            encoded = encode_label(text).numpy()
            if len(encoded) > MAX_LABEL_LENGTH:
                continue
            labels[i, :len(encoded)] = encoded
            label_lengths[i] = len(encoded)

        input_lengths = np.ones((self.batch_size, 1), dtype=np.int32) * (IMG_WIDTH // 4)
        return {
            "image_input": images,
            "label_input": labels,
            "input_length": input_lengths,
            "label_length": label_lengths
        }, np.zeros((self.batch_size,))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.data = [self.data[i] for i in self.indices]



# Load and Validate .npy Files
def load_npy_data(images_path, labels_path):
    images = np.load(images_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    dataset = []
    skipped = 0
    unk_labels = 0

    for img_path, label in zip(images, labels):
        # Validate label
        if not isinstance(label, str) or not label or not all(c in CHARSET for c in label):
            skipped += 1
            continue
        # Check for [UNK] tokens in encoded label
        encoded = encode_label(label).numpy()
        if char_to_num.get_vocabulary().index("[UNK]") in encoded:
            unk_labels += 1
            skipped += 1
            continue
        # Normalize path for Windows
        img_path = os.path.normpath(img_path)
        # Check if image file exists
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            skipped += 1
            continue
        # Validate image
        img = preprocess_image(img_path)
        if img is None:
            print(f"Invalid image (processing failed): {img_path}")
            skipped += 1
            continue
        dataset.append((img_path, label))

    print(f"Loaded {len(dataset)} valid samples, skipped {skipped} invalid samples "
          f"(including {unk_labels} with [UNK] tokens).")
    if dataset:
        print("Sample labels:", [label for _, label in dataset[:5]])
    return dataset




# Visualize Predictions
def decode_prediction(pred, max_len=MAX_LABEL_LENGTH):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    return tf.strings.reduce_join(num_to_char(results), axis=1).numpy()





# 2. Prepare model input
def prepare_input(img_tensor):
    return tf.expand_dims(img_tensor, axis=0)  # add batch dimension

# 3. Make prediction and decode
def predict_image(model, img_path):
    img_tensor = preprocess_image(img_path)
    model_input = prepare_input(img_tensor)
    prediction = model.predict(model_input)
    decoded = decode_prediction(prediction)
    
    # Visualize
    plt.imshow(tf.squeeze(img_tensor), cmap='gray')
    plt.axis('off')
    plt.title("Predicted: " + decoded[0].decode('utf-8'))
    plt.show()

    return decoded[0].decode('utf-8')





# --- Config ---
IMG_WIDTH = 128
IMG_HEIGHT = 32



def preprocess_image(img_path, augment=False):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0

        if augment and random.random() < 0.5:
            angle = random.uniform(-5, 5)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=1.0)
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0.0, 1.0)

        img = np.expand_dims(img, axis=-1)
        return img
    except:
        return None

def prepare_input(img_tensor):
    return tf.expand_dims(img_tensor, axis=0)

def predict_image(model, img_path):
    img_tensor = preprocess_image(img_path)
    if img_tensor is None:
        return "[Error: Image Not Read]"
    model_input = prepare_input(img_tensor)
    prediction = model.predict(model_input)
    decoded = decode_prediction(prediction)
    return decoded[0].decode('utf-8')

# --- Word Extraction and Prediction ---
def extract_and_predict_words(image_path, model):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Group into lines
    lines = []
    tolerance = 20

    for box in bounding_boxes:
        x, y, w, h = box
        cy = y + h // 2
        added = False
        for line in lines:
            ly = line[0][1] + line[0][3] // 2
            if abs(ly - cy) < tolerance:
                line.append(box)
                added = True
                break
        if not added:
            lines.append([box])

    for line in lines:
        line.sort(key=lambda b: b[0])
    lines.sort(key=lambda line: min(b[1] for b in line))

    sorted_boxes = [box for line in lines for box in line]

    os.makedirs("words", exist_ok=True)
    final_text = ""
    count = 0

    for x, y, w, h in sorted_boxes:
        if w > 50 and h > 30:
            word_img = img[y:y+h, x:x+w]
            word_path = f"words/word_{count}.png"
            cv2.imwrite(word_path, word_img)

            # Predict text
            predicted_text = predict_image(model, word_path)
            print(f"Word {count}: {predicted_text}")
            final_text += predicted_text + " "
            count += 1

    print(f"\nFull sentence:\n{final_text.strip()}")




def preprocess_image_array(img_array, augment=False):
    try:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0

        if augment and random.random() < 0.5:
            angle = random.uniform(-5, 5)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=1.0)
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0.0, 1.0)

        img = np.expand_dims(img, axis=-1)
        return img
    except:
        return None



def predict_from_array(model, word_img_array):
    img_tensor = preprocess_image_array(word_img_array)
    if img_tensor is None:
        return "[Error: Preprocessing failed]"
    model_input = prepare_input(img_tensor)
    prediction = model.predict(model_input)
    decoded = decode_prediction(prediction)
    return decoded[0].decode('utf-8')

# --- Extract word boxes and run prediction ---
def extract_and_predict_words(image_path, model):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Group into lines
    lines = []
    tolerance = 20

    for box in bounding_boxes:
        x, y, w, h = box
        cy = y + h // 2
        added = False
        for line in lines:
            ly = line[0][1] + line[0][3] // 2
            if abs(ly - cy) < tolerance:
                line.append(box)
                added = True
                break
        if not added:
            lines.append([box])

    for line in lines:
        line.sort(key=lambda b: b[0])
    lines.sort(key=lambda line: min(b[1] for b in line))

    sorted_boxes = [box for line in lines for box in line]

    final_text = ""
    count = 0

    for x, y, w, h in sorted_boxes:
        if w > 50 and h > 30:
            word_img = img[y:y+h, x:x+w]  # crop directly
            predicted_text = predict_from_array(model, word_img)
            print(f"Word {count}: {predicted_text}")
            final_text += predicted_text + " "
            count += 1

    final_text = final_text.strip().replace('[UNK]', '')

    print("\nFinal sentence:\n" + final_text.strip())

def run(image_path):
    # # --- Run the complete pipeline ---
    image_path = image_path.replace("\\", "/").replace("\"", "")
    img = cv2.imread(image_path)
    prdicted_text = predict_from_array(model=model, word_img_array=img)
    final_text = prdicted_text.strip().replace('[UNK]', '')
    print("Final Text: ", final_text)


