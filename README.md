# Brain-Machine Interface for Robotic Control 🧠➡️🤖

This project aims to build a complete Brain-Machine Interface (BMI) system to control a robot using electroencephalography (EEG) signals. The core of the project involves decoding motor imagery (MI) brainwaves—the signals generated when a person imagines moving their left or right hand—to produce control commands.

The pipeline covers the full spectrum of a BCI project: from raw signal preprocessing with MNE-Python to training deep learning models (ANN/CNN) with TensorFlow/Keras, with the end goal of real-time robotic integration.

## Project Pipeline

The workflow is structured as follows:

`Raw EEG Data (.edf) -> MNE Preprocessing (Filtering, ICA) -> Labeled Epochs -> Model Training (TensorFlow) -> Robot Command`

## Key Features

* **EEG Preprocessing:** A robust pipeline for cleaning raw EEG data, including band-pass filtering and automated artifact removal using Independent Component Analysis (ICA).
* **Motor Imagery Classification:** Trains deep learning models to classify between left and right imagined fist movements.
* **Data Source:** Utilizes the well-established **PhysioNet EEG Motor Movement / Imagery Dataset**.
* **Deep Learning:** Implements and evaluates both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) for the classification task.

## Getting Started

### Prerequisites

* Python 3.10 (for GPU support with TensorFlow)
* An NVIDIA GPU with CUDA and cuDNN installed (for the `.venv-gpu` environment)

### Installation and Setup

Follow these steps in your terminal to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Littnatenate/BMI-Robotic-Control.git](https://github.com/Littnatenate/BMI-Robotic-Control.git)
    cd BMI-Robotic-Control
    ```

2.  **Create the Virtual Environment:**
    This project includes two potential environments. For GPU-accelerated training, create and use the `.venv-gpu`.

    * **To create the GPU environment (recommended):**
        ```powershell
        py -3.10 -m venv .venv-gpu
        ```

3.  **Activate the Environment:**
    You must activate the environment in your terminal before installing packages or running the script.

    * **To activate the GPU environment:**
        ```powershell
        .\.venv-gpu\Scripts\Activate.ps1
        ```

4.  **Install Required Libraries:**
    This command reads the `requirements.txt` file and installs all necessary packages like TensorFlow and MNE.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main preprocessing pipeline is located in `src/main.py`. This script is configured to download the necessary data, preprocess it for a single subject, and prepare it for model training.

1.  Make sure your virtual environment is active.
2.  Run the main script from the project root directory:
    ```bash
    python src/main.py
    ```

This will run the full preprocessing pipeline for a single subject and output the final shape of the data ready for machine learning.