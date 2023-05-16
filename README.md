# reservoir-simulation
**UNDER DEVELOPMENT! WILL PROVIDE UPDATES SOON**
The second function (reservoir optimization), the GUI, and the dataset is underdevelopment.
Will be creating a custom model and a custom GUI in the near future.

**CONTEXT**: Petroleum Engineering, Oil Reservoir Engineering

**1.0 PROJECT DESCRIPTION:**
This Python project builds on my Reservoir Engineering project: Compilation of a systematic framework for identification of crucial parameters and interactions affecting ultimate oil recovery of heavy oil reservoirs based on design of experiments (DOE).

It uses a dataset of 54 datapoints that I obtained during my work experience and also as final project (mentioned above) that I delivered to a committee. This project was created after the initial delivery as a personal project to explore and see if Machine Learning could be used to predict reservoir parameters (compared with traditional methods).

**2.0 DISCLAIMER:**
Unfortunately, since the data was obtained during my employment at a certain facility, I am not able to share it publicly. Therefore, this project acts as a proof of concept. The dataset can be replaced by your own datapoints. 

To combat this, I will create and upload the precompiled model itself in the near future.

**3.1 DATA**
We use an ANN model to predict the behavior of Oil Reservoirs based on some provided reservoir attributes.

The dataset used contains 12 columns:

*inputs
**a**: permeability factor representing the permeability changes due to different production scenarios (INPUT

**Q**: oil production rate from the production well (INPUT)

**INJ**: water injection rate of the injection wells at the corners of the reservoir

**m:** oil viscosity factor depending on the production scenario(INPUT)

**b**: Interfacial tension factor relating to the surface phenomena occurring at water-oil confrontation(INPUT)

**D**: the angle perpendicular to the horizon plane that the production well has been drilled (INPUT)

*outputs

**a/m**: the division of a over m for simplicity (INPUT)

**URF**: ultimate recovery factor of oil in fraction (OUTPUT)

**P**: reservoir pressure (OUTPUT)

**J**: the productivity index of the production well (OUTPUT)

**GOR**: gas oil ratio of the production well (OUTPUT)

**WC**: water cut ratio of the production well (OUTPUT)


**3.2 MODEL**
One ANN model is constructed (so far) with the following attributes (additional information above)

Dense Input Layer (X): 
- 'a'
- 'Q'
- 'INJ'
- 'm'
- 'b'
- 'D'

Outputs (y):
- 'URF (fr.)'
- 'P (psi)'
- 'J (STBD/psi)'
- 'GOR (Mscf/STB)'
- 'WC (fr.)'

 
Activation function:
- ReLU (Rectified Linear Unit)

The model is compiled using the Mean Squared Error (MSE) loss function and the Adam optimizer. The Mean Absolute Error (MAE) is used as a performance metric during training.

**4.0 RESULTS**
**under development**
Console after 100 epochs:
Epoch 100/100
34/34 [==============================] - 0s 848us/step - loss: 0.0072 - mae: 0.0639 - val_loss: 0.0076 - val_mae: 0.0682
Test MAE: 0.08348092436790466
1/1 [==============================] - 0s 40ms/step

Loss:
![image](https://github.com/MSF9119/reservoir-simulation/assets/133431610/ca1726dd-d2ee-4a66-aee9-46c1d94b46a1)

Predicted Outputs:
![image](https://github.com/MSF9119/reservoir-simulation/assets/133431610/d8147a77-bc79-409e-b8ed-efc945022c82)



**5.0. TO RUN THE SCRIPT:**
1- Create a new empty folder.
2- Create (I'm sorry! will upload the model in the future) your own dataset labeled "datapoints2.csv" inside the folder.
3- Download and put the Python code inside the same folder.
4- Run the code

**6.0. MODES AND USAGE**
UNDER DEVELOPMENT
Function 1: So far, the code will display the performance metrics of the model and allow the user to enter all the input parameters and the app will then predict how an oil reservoir will behave (by providing its attributes) given the input conditions (it basically simulates the reservoir)

Future functions:
Function 2: Optimize reservoir parameters based on dependant conditions (how should my input material be for my reservoir to be optimized?)
Function 3: Optimize input parameters based on desired reservoir conditions (example: which input material conditions to use to have an oil reservoir have a specific pressure)
