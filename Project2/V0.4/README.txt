Project deliverable is a zip called project_02.zip and contains our implementation of
an Adaboost algorithm that uses our weak learner implementation.

When unzipped, main folder 'adaboost' contains the following sub-folders and files:
adaboost
├── adaboost.py
├── adaboost-test-24.txt
├── adaboost-train-24.txt
├── common
│   ├── algorithm.py
│   ├── plot_library.py
│   └── weak_classifier_base.py
├── implementations
│   ├── custom_weak_classifier.py
│   └── sklearn_weak_classifier.py
├── README.txt
└── sklearn
    ├── sklearn_adaboost.py
    └── sktlearn_adaboost.ipynb
3 directories, 11 files

The README.txt is this file.

We have provided support to explore 
Note all cmds below are run from inside 'adaboost' main folder.
To run our implementation, from cmd line enter:
python adaboost.py

Out of curiosity, to run the sklearn adaboost algorithm with our weak learner implementation, then from cmd line enter:
cp sklearn/sklearn_adaboost.py .
python sklearn_adaboost.py

The full sklearn implementation, ie sklearn adaboost algorithm with sklearn decision tree stump, can also be run
bu editing the sklearn_adaboost.py file and selecting that learner (as per comment in line 33 of the code)

A final combination, ie our adaboost algorithm using the sklearn decision tree stump can also be run together, by
editing the adaboost.py file and selecting that learner (as per comment in line 48 of the code)









