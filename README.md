# MyDigitNet
A simple neural network app made for guessing hand-written digits.
![screenshot2](/screenshots/example2.png)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
  
## Features

- **Digits Classification**: High accuracy digit recognition using a custom CNN architecture.
- **Customizable Training**: Easily adjust model parameters such as learning rate, batch size, and number of epochs.
- **Model Evaluation**: Visualize performance metrics and confusion matrices to assess model accuracy.
- **Pre-trained Models**: Start with pre-trained models or train from scratch.

## Installation

1. Clone repository to your pc
   
     ```
     git clone https://github.com/TAskMAster339/MyDigitNet
     ```
     
2. Make sure you have [cmake](https://cmake.org/) and [MinGW](https://www.mingw-w64.org/)
   
3. Go to the repository directory
   ```
   cd MyDigitNet
   ```
   
4. Use cmake to create the build directory
   
   ```
   cmake -G "MinGW Makefiles" -B ./build/
   ```
   
   *-G MinGW Makefiles* force cmake to use MinGW
5. Build the executable file

   ```
   cmake --build ./build/
   ```
   
6. Now you can go to build directory and open *MyDigitNet.exe* \
   or write in cmd

   ```
   start build/MyDigitNet.exe
   ```

## Usage
![screenshot1](/screenshots/example1.png)
1. Check src/Config.txt file. First of all Neural network will read this file and set up its configuration.
   Deafult configuration is
   ```
   NetWork 3
   784
   256
   10
   ```
   NetWork is a key-word, which defines the config file. 3 is a number of layers. 784, 256, 10 are numbers of neurons per layer.
2. In the open console window first of all you will see the settings of activation function. You can choose *Sigmoid*, *reLU*, *tanh*.
3. When the program will ask you to train your Neural network. By default the neural network is already trained and you can easyly skip this part. (Neural network will load weight from src/Weights.txt). But you can train network again with another training configuration.
4. After training/or not you can test your Neural network, or skip this part.
5. In the end, if you made any mistake, you can restart training/testing process.
6. Now you have successfully set up your Neural network and it is ready to use. On the screen you can see a black sqare, where you can draw your digits. If you make any mistake, you can press "c" to clean the screen. When your picture is ready, you can click right mouse button to send your drawing to NeuralNetwork.
7. Finally, you see network prediction in the console.

 ## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.
