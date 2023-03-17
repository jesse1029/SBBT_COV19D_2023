## 2023 COVID-19-CT-Classification
The competition includes two types of frameworks for the scripts. Except for the difference in network framework, all data preprocessing is the same for both.

==The following script assumes that the data and paths are correct. Please make sure that the data and paths are correct before running the script.==
***
Run the ```00-chih-fold-4-covid-df(base+challenge).ipynb``` script which treats all train and validation sets as cross-validation. Use it as needed.
Run the ```01-preprocess.py``` script to perform cropping on each slice.
Run the ```02-get_slice_range.ipynb``` script to obtain the range for each CT scan.
Run the ```03-1-slice_filter_code.ipynb``` script to handle exceptional images, such as excluding misaligned axis data, modifying paths, and constructing convenient CSV files for training or testing.
***

### mca-nfnet
Run the ```submit0_eca_nfnet_l0_cv_training.py``` script to obtain basic results.
Run the code under ```threshold``` to obtain the optimal cut-off point for validation.
### eff-conv-mix
Run the ```04-0-2dcnn_model_dp.py``` script to obtain basic results.
Run the ```05-2dcnn_get_embedding.ipynb``` script to obtain the embedding features of each slice.
Train the second-stage model using the ```06-2step_try.ipynb``` script.
Perform initial verification using the ```07-2step_inference.ipynb``` script.