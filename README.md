# EEGEmotionClassificationGAMEEMO
CSC 570 Final Project 


Emotion recognition from Electroencephalogram(EEG) signals have become more widespread for affective computing and constructing insights into emotional states during interactive experiences. These insights could prove useful for improving user experience, understanding true human reactions, and capturing neural relationships which could prove invaluable for the gaming, entertainment, and marketing industry. This project utilizes the results from the GAMEEMO dataset generation paper to classify EEG data of subjects playing four different games that correspond to four classes in the Arousal-Valence model. We attempt a reproduction of the paper’s classification model by replicating their entire pipeline. To investigate the classification abilities further, we conduct our own investigation into the dataset and build out our own pipeline. Our novel contributions include validating the paper’s methods, exploratory data analysis on the dataset, and using our own models to predict emotion states using EEG data. 
Emotion recognition from Electroencephalogram (EEG) signals has become increasingly important in affective computing, as it provides a foundation for understanding emotional states during interactive experiences. Such insights can improve user experience, reveal true human reactions, and capture neural patterns that are valuable in the gaming, entertainment, and marketing industries.

In this project, we attempt to reproduce the methods from “Database for an emotion recognition system based on EEG signals and various computer games — GAMEEMO” by Alakuş et al. (2020) [1], which introduces the GAMEEMO dataset and proposes a complete preprocessing, feature extraction, and classification pipeline for emotion recognition. Using the GAMEEMO dataset, we replicate the paper’s workflow to classify EEG recordings from subjects playing four video games corresponding to distinct quadrants of the Arousal–Valence model.

To extend this work, we conduct independent exploratory data analysis (EDA) and develop additional machine learning pipelines to evaluate classification performance beyond the original approach. Our contributions include validating aspects of the paper’s methodology, analyzing the dataset’s statistical and spectral properties, and implementing our own models to predict emotional states from EEG signals.

[1]  Alakuş, T. B., Gönen, M., & Türkoğlu, I. (2020). Database for an emotion recognition system based on EEG signals and various computer games — GAMEEMO. Biomedical Signal Processing and Control, 60, 101951



# Instructions

To run SAM score extraction
1. Load the .zip file from kaggle and save
2. run ```python extract_SAM_scores.py```

To run feature extraction from paper
1. update list of features in feature_extraction function
2. run ```python feature_extraction_1```
 
To run windowed feature extraction 
1. Change window_size and window_overlap in main
2. Run ```python feature_extraction_2```

To run the model training pipeline
1. Download the CSC570FinalCode.ipynb
2. Ensure that you have the right files provided in this repo (bandpower_features folder + final_SAM_Scores.pdf + simple_features_paper.csv) 
3. <img width="2402" height="1031" alt="image" src="https://github.com/user-attachments/assets/a5330b2c-cbbd-472a-b42f-dc2db825eb4e" />
