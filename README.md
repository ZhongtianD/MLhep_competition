# MLhep_competition

Works in the same way as the baseline code provided.

Regression model is simpcov. Classification model is Resnet2.
Please change the modle names respectively in config, train.py, report.py, and generate_submissions.py

I do not have the weights of the best models I have under this repository.

The data preparation is a little different.
I seperate test class from the default training set and call them the val set and use that as validation.
The rest from the training set are used for training.
