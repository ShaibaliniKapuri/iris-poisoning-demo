# Data Poisoning Attack Simulation on the Iris Dataset 🧪

This repository demonstrates a classic **data poisoning attack** on a machine learning model. The script trains a Support Vector Machine (SVM) on the famous Iris dataset and shows how its performance degrades when the training data is "poisoned" at various levels.

The primary goal is to illustrate the vulnerability of machine learning models to corrupted training data and to provide a clear, executable example of this security threat.



***

## 🚀 Key Concepts Demonstrated

* **Data Poisoning**: Intentionally corrupting training data to compromise a model's performance.
* **Attack Vector**: The script uses a combination of *feature corruption* (replacing feature values with random noise) and *label flipping* (changing the class label).
* **Model Training & Validation**: A Support Vector Machine (`SVC`) is trained on the (potentially poisoned) data and then evaluated against a clean, untouched validation set.
* **Impact Analysis**: The script measures and displays the drop in model accuracy as the percentage of poisoned data increases from 0% to 50%.

***

## 🛠️ How to Run the Experiment

You can easily run this experiment in a cloud environment like GCP Cloud Shell or on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/](https://github.com/)<Your-Username>/iris-poisoning-demo.git
    cd iris-poisoning-demo
    ```

2.  **Install Dependencies**
    The project requires `scikit-learn` and `numpy`.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt` file, you can install them directly: `pip install scikit-learn numpy`)*

3.  **Execute the Script**
    ```bash
    python3 poison_iris.py
    ```

***

## 📊 Expected Outcome

When you run the script, it will print the validation accuracy of the model for each poisoning level. The output clearly shows the degradation of the model's performance.
