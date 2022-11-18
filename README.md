# DomainClassification
Classifying translation content into different domains using RoBERTa. 

`data_preprocessing.py` preprocesses and extracts information from XML files.
To start fine-tuning a RoBERTA model, run

`python create_roberta_model.py' --train_file <train_file> --val_file <validation_file>`

Here is my medium article that describes the challenges and results of this vertical classification task: https://medium.com/p/ea5bc434a49a
