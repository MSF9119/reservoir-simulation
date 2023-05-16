# This Python project builds on my Reservoir Engineering project:
# Compilation of a systematic framework for identification of crucial parameters and interactions affecting ultimate oil recovery of heavy oil reservoirs based on design of experiments (DOE).

# It uses a dataset of 54 datapoints that I obtained during my work experience and also as final project
# This project was created as a personal project to explore and see if Machine Learning could be used to predict reservoir parameters (compared with traditional methods).

# We will design a neural network model to predict the parameters using the values I experimentally obtained from a functioning
# oil reservoir.
# We need to import the following libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# This is a library to use for GUI
import tkinter as tk

# We read the datapoints first
data = pd.read_csv('datapoints2.csv')

# Split the dataset into input (X) and output (y) variables based on columns
X = data[['a', 'Q', 'INJ', 'm', 'b', 'D']].values
y = data[['URF (fr.)', 'P (psi)', 'J (STBD/psi)', 'GOR (Mscf/STB)', 'WC (fr.)']].values

# Scaling the extracted data
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X = X_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

# Split the dataset into training and test sets (I always use 80 - 20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now the fun part, we construct the model with a dense input layer
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(6,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # compile
history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2)

# And finally evaluate!
_, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Test MAE: {test_mae}')

# to plot the performance metrics, such as training and validation loss:
# We create the first figure for training and validation losses
fig1 = plt.figure()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Now the testing phase
# We predict the output for the test set
y_pred = model.predict(X_test)


# For performance plots, for each predicted output, plot the target output vs predicted output
# Figure 2:
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 5 columns: 5 outputs
for i in range(5):
    row = i // 3
    col = i % 3
    axs[row, col].scatter(y_test[:, i], y_pred[:, i])
    axs[row, col].plot([0, 1], [0, 1], 'r--')
    axs[row, col].set_title(f'Output {i+1}')
    axs[row, col].set_xlabel('Target')
    axs[row, col].set_ylabel('Predicted')
    axs[row, col].set_xlim(0, 1)
    axs[row, col].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Save the model to be used in the future
model.save('trained_model.h5')


# ********************************  GUI  *****************************
# UNDER DEVELOPMENT: THE SECOND FUNCTION
# Create main window and set its title.
root = tk.Tk()
root.title("Meysam's Petroleum Attribute Advisor")

labels_text=["a", "Q", "INJ", "m", 'b', 'D']
entries=[]

# to predict a model based on simulation
for i,text in enumerate(labels_text):
	label=tk.Label(root,text=text+":")
	entry=tk.Entry(root,width=10)
	label.grid(row=i,column=0,padx=(20,))
	entry.grid(row=i,column=1,padx=(20,))
	entries.append(entry)



# GUI SETUP
import tkinter as tk


# FUNCTION 1: PREDICT/SIMULATE WELL based in input condition
def show_predict_inputs():
    input_window = tk.Toplevel(root)
    input_window.title("Enter Inputs")

    for i, text in enumerate(input_label_texts):
        label = tk.Label(input_window, text=text + ":")
        entry = tk.Entry(input_window, width=10)
        label.grid(row=i, column=0, padx=(10,))
        entry.grid(row=i, column=1, padx=(10,))

        # store all enteries
        input_entries.append(entry)

    predict_button = tk.Button(input_window, text="Predict Output!", command=predict)
    predict_button.grid(row=len(labels_text), columnspan=2, pady=(15))

    output_label = tk.Label(input_window, textvariable=result)
    output_label.grid(row=len(labels_text) + 1, columnspan=2, pady=(10))


# Function 2: **** UNDER DEVELOPMENT ***
#def show_optimize_all_inputs():


def predict():
    # Get input values from Entry widgets and convert them into a list of floats
    inputs = []
    for entry in input_entries:
        try:
            value = float(entry.get())
            inputs.append(value)
        except ValueError:
            result.set("Please enter valid numbers for all input fields.")
            return

    # Prepare new inputs for prediction (reshape it into a 2D array with one row)
    input_data = np.array(inputs).reshape(1, -1)

    # Scale new inputs using same scaler used during training process
    scaled_input_data = X_scaler.transform(input_data)

    # Predict outputs using trained model object we have loaded before.
    predicted_outputs = model.predict(scaled_input_data)

    original_scale_predicted_outputs = y_scaler.inverse_transform(predicted_outputs)
    result.set("Predicted Outputs: " + ", ".join([str(round(val, 4)) for val in original_scale_predicted_outputs[0]]))
    # Create a popup window to display predicted outputs
    output_window = tk.Toplevel(root)


    output_window.title("Predicted Outputs")
    predicted_output_text = "Predicted Outputs: " + " , ".join([str(round(val, 4)) for val in original_scale_predicted_outputs[0]])

    output_label = tk.Label(output_window, text=predicted_output_text)
    output_label.pack(padx=(20), pady=(20))



root = tk.Tk()
root.title('Neural Network Prediction')

main_frame = tk.Frame(root, pady=(20,))
main_frame.pack()

button_01 = tk.Button(main_frame, text='Predict Output', command=show_predict_inputs)
# DONT CLICK THIS ONE, UNDER DEVELOPMENT
#button_02 = tk.Button(main_frame, text='Optimize all Inputs', command=show_optimize_all_inputs) REMOVE COMMENT


button_01.grid(row=0, column=0, padx=(10))
#button_02.grid(row=0, column=1, padx=(10)) #REMOVE COMMENT


input_label_texts = ["a", "Q", "INJ", "m", 'b', 'D']
input_entries = []

result = tk.StringVar()

root.mainloop()