# Loan Approval Prediction with Machine-Learning
Author: Yong-Sung Masuda  
Date: December 20, 2024
## Purpose
Loan applications are approved or denied by financial institutions based on a set of disclosed criteria (credit history length, loan amount, loan intent, etc.) but with undisclosed decision-making processes. The result of a loan application likely varies based on the institution it was submitted to and the loan officer tasked with its review. Since the process is not strictly defined, predicting the outcome of a loan application is a problem well suited to a machine-learning solution. A well-trained prediction model may be used by financial institutions for various purposes, including application triage, automatic application approvals, and targeted marketing campaigns. Such a prediction model may also be of service to potential applicants in order for them to gauge what outcome to expect and perhaps adjust their loan application before submission.  
## Dataset
The dataset used for this model was sourced from Kaggle.com, specifically from season 4, episode 10 of their playground series. This dataset was generated from a deep learning model trained on a loan approval prediction dataset with similar feature distributions. The dataset includes a labeled training set with 58,645 samples and an unlabeled test set of 39,098 samples. The training set is divided into three subsets:  
-	Training set of 41,051 samples (70%)  
-	Validation set of 8,797 samples (15%)  
-	Private test set of 8,797 samples (15%)  
The training set is used to fit models to the data during the model selection phase. The validation set is used to evaluate the models for selection during hyperparameter tuning and is also used for early stopping. The private test set is used to ensure the selected model is not overfit to the training and validation sets, and is also used to compute additional performance metrics.  
 
## Data Preprocessing
The parameters of each data sample are preprocessed as either categorical or numerical. Categorical parameters are one-hot encoded and numerical parameters are regularized to be a floating-point decimal between 0 and 1.
The categorical parameters are:  
-	Person home ownership (rent, own, or mortgage)  
-	Loan intent (education, medical, personal, etc.)  
-	Loan grade (A, B, C, D, or E)  
-	Person default on file (yes or no)  

The numerical parameters are:  
-	Person age  
-	Person income  
-	Loan amount  
-	Loan interest rate  
-	Loan percent of income
-	Person credit history length  
-	Person employment length  
## Model Architecture and Hyperparameter Tuning
The neural network implemented has two hidden layers each with ReLU activation and dropout regularization. The sigmoid activation function is applied to the final layer, resulting in an output between 0 and 1 indicating the probability of loan approval. Bayesian optimization of hyperparameters is implemented with the Optuna optimization framework. The hyperparameters tuned and their range limits are:  
-	Sizes of each hidden layer (16 to 256 neurons)  
-	Dropout rate (10% - 50%)  
-	Learning rate (1e-5 – 1e-1)  
-	Batch size (16 – 256 samples)  
A total of 20 trials were conducted, with each trial trained for a maximum of 100 epochs. Early stopping was implemented with a patience of 7 epochs. The trial that yielded the best results was trial 3. The hyperparameters from this trial are used to train the final model with all of the available data (training, validation, and private test set combined) for 100 epochs with early stopping (best results were achieved after epoch 87). The complete results of the hyperparameter tuning are presented in following table

| Trial | Hidden Size 1 | Hidden Size 2 | Dropout Rate        | Learning Rate          | Batch Size | Accuracy           |
| ----- | ------------- | ------------- | ------------------- | ---------------------- | ---------- | ------------------ |
| 0     | 130           | 153           | 0.3456974698382512  | 1.0803515819378067e-05 | 222        | 0.9324769807889053 |
| 1     | 34            | 201           | 0.11772388345150527 | 0.020560841144176122   | 256        | 0.9424803910424008 |
| 2     | 71            | 239           | 0.38768063855877366 | 0.02206433490909725    | 48         | 0.9395248380129589 |
| 3     | 50            | 48            | 0.11866136533720831 | 0.0013796983303193278  | 187        | 0.9446402182562237 |
| 4     | 223           | 248           | 0.2906787967558432  | 4.34038679395373e-05   | 25         | 0.9416846652267818 |
| 5     | 135           | 98            | 0.31040889700087904 | 6.718642268852034e-05  | 129        | 0.9442991929066727 |
| 6     | 238           | 233           | 0.45224046794929407 | 0.027945826639996854   | 217        | 0.9432761168580198 |
| 7     | 182           | 40            | 0.3826868950716771  | 2.4036715551065972e-05 | 166        | 0.940093213595544  |
| 8     | 134           | 226           | 0.24973094211687005 | 0.09740154466600491    | 17         | 0.8579061043537569 |
| 9     | 223           | 199           | 0.4467209731691426  | 1.619417728447839e-05  | 148        | 0.9411162896441969 |
| 10    | 22            | 19            | 0.10561677564458889 | 0.0013331272034396789  | 102        | 0.9441855177901557 |
| 11    | 83            | 72            | 0.20106089877107042 | 0.0003655733603615575  | 105        | 0.9424803910424008 |
| 12    | 91            | 95            | 0.2013369555395409  | 0.0006835233095759001  | 166        | 0.9422530408093668 |
| 13    | 183           | 115           | 0.1820766123526748  | 0.00014280156675564504 | 105        | 0.9414573149937479 |
| 14    | 171           | 63            | 0.30699153108337296 | 0.002487720771921916   | 194        | 0.9421393656928498 |
| 15    | 60            | 139           | 0.2561082139302825  | 0.0035593179562429327  | 134        | 0.9433897919745368 |
| 16    | 109           | 93            | 0.16345730715226756 | 0.00014549177855951824 | 80         | 0.9431624417415028 |
| 17    | 49            | 53            | 0.24258949326901832 | 9.796457409700543e-05  | 187        | 0.9432761168580198 |
| 18    | 156           | 161           | 0.3438448586324775  | 0.006288754692924151   | 139        | 0.9445265431397067 |
| 19    | 158           | 171           | 0.48499623840157924 | 0.007074120844630719   | 253        | 0.9421393656928498 |
  
## Evaluation 
The best model selected from hyperparameter tuning achieves a prediction accuracy of 94.46% on the validation set (used for model selection) and an accuracy of 94.57% on the private test set. The similar results achieved on the validation set and private test set confirms that the model is not overfit to the training and validation data, and is able to extrapolate. Additional evaluations are performed on the results from the private test set.  

![image](/images/loan_approval_figure_1.png)
  
These graphics are not available for the final model since it is trained on the entire training set and the labels for the test set are unavailable. To evaluate the final model, predictions on the unlabeled test set are submitted to Kaggle where they are segmented into two sets in order to provide two AUROC scores, one being displayed until the competition deadline, and the other being published after. The pre-deadline score achieved by the final model is 0.93740 and the post-deadline score is 0.93313.

## Acknowledgements
The code for this project was developed while consulting various AI chatbots powered by Large Language Models (LLMs), including Claude Sonnet 3.5, Gemini 1.5 Flash, and ChatGPT 3.5. While the models did not produce directly usable code, they were instrumental in drafting code and providing usage examples for Python library functions. The human contributors behind the training data for these models, likely users of platforms such as GitHub and Stack Overflow, deserve recognition for their indirect support.