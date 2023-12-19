# Leaf classification

This project is a part of the IFT712 course at Université de Sherbrooke. It was made in collaboration with [Alexandre Theisse](https://github.com/AlexTheisse) and [Tom Sartori](https://github.com/tom-sartori).

The goal of this project is to classify leaves from features extracted from their images and from the images themselves using several classification algorithms available in the `scikit-learn` library. The dataset used is the [Folio Leaf Dataset](https://www.kaggle.com/c/leaf-classification/data) from Kaggle.

## Contributors

|          Nom          | Matricule  |   CIP    |                                        Mail                                         |
|:---------------------:|:----------:|:--------:|:-----------------------------------------------------------------------------------:|
|   Alexandre Theisse   | 23 488 180 | thea1804 |     [alexandre.theisse@usherbrooke.ca](mailto:alexandre.theisse@usherbrooke.ca)     |
| Louis-Vincent Capelli | 23 211 533 | capl1101 | [louis-vincent.capelli@usherbrooke.ca](mailto:louis-vincent.capelli@usherbrooke.ca) |
|      Tom Sartori      | 23 222 497 | sart0701 |           [tom.sartori@usherbrooke.ca](mailto:tom.sartori@usherbrooke.ca)           |

## Getting Started

### Requirements
Install the requirements using the following command:
```bash
pip install -r requirements.txt
```

### Running the code

The code is split into 4 notebooks:
- `features_classification.ipynb`: Classification using only the features extracted from the images
- `features_hyperparam_search.ipynb`: Hyperparameter search for the features classification
- `image_classification.ipynb`: Classification using only the images
- `image_hyperparam_search.ipynb`: Hyperparameter search for the image classification
