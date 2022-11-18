# DomainClassification
Classifying translation content into different domains using RoBERTa. 

`preprocess_data.py` preprocesses and extracts information from XML files.
To start fine-tuning a RoBERTA model, run

`python create_roberta_model.py' --train_file <train_file> --val_file <validation_file>`
