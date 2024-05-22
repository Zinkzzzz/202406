import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

data_dir = 'data'

texts = []
labels = []
for file_name in os.listdir(data_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
            texts.append(text_content)
            labels.append(file_name.split('.')[0])

if not texts:
    print("No text sequences found. Exiting program.")
    exit()

num_classes = len(set(labels))

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

max_length = max([len(seq) for seq in sequences])

vocab_size = len(tokenizer.word_index) + 1

padded_sequences = sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

label_encoder = LabelEncoder()
label_seq = label_encoder.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(padded_sequences, label_seq, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(optimizer=adam_optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)


model.fit(X_train, y_train, epochs=500, validation_split=0.2, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")