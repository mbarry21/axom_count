import keras
import pandas as pd
import matplotlib.pyplot as plt

model = keras.models.load_model("vile_counter.h5")
history = pd.read_parquet("vile_counter_results.parquet")
print(history.keys())

# Display accuracy plot
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')


# Display loss plot
plt.subplots()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# TODO: Display some sample predictions
