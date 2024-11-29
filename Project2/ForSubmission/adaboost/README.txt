Project deliverable is a zip called project_02.zip and contains our implementation of
an Adaboost algorithm that uses our weak learner implementation ie 2 software components.

Project Team
Qinyuan Liu           20137095 
Eamon Moloney         8457077 
Ibrahim Saana Aminu   25381993 
Des Powell            9513833 
Terence Coffey        15223124 


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
An explanation of the purpose of each file is provided in the documentation.

Execution
Note all cmds below are run from inside 'adaboost' main folder.
To run our custom implementation, from cmd line enter:
python adaboost.py

In addition, We have provided support to explore a little more, if curious, which was developed to help us design and test our 2 software components seperatly.
As a result, the following can be run, if desired:
    To run the sklearn adaboost algorithm with our custom weak learner implementation, then from cmd line enter:
    cp sklearn/sklearn_adaboost.py .
    python sklearn_adaboost.py

    To run the full sklearn implementation, ie sklearn adaboost algorithm with sklearn decision tree stump, edit the sklearn_adaboost.py file 
    and select that learner (as per comment in line 39 of the code) - be sure to run sklearn_adaboost.py from the adaboost directory

    Finally, to run our custom adaboost algorithm using the sklearn decision tree stump, editing the adaboost.py file and selecting that 
    learner (as per comment in line 47 of the code)


Output
When python adaboost.py is run, the following 5 items are output:
    # 1 - TEST ACCURACY AFTER Ntest ITERATIONS - using specified best Ntest  
    # 2 - MEASURE TRAIN & TEST ACCURACY OVER Ntest plus a bit more iterations (so you can see when )   
    # 3 - GENERATE DECISION BOUNDRY DATA - using training data
    # 4 - GENERATE CONTOUR DATA - using test data
    # 5 - PLOTS - plots of train v test accuracy, decision boundry, and contour, all rolled into a single figure.

Dcoumentation includes various pictures, etc and will indicate how long each of the above steps take.










