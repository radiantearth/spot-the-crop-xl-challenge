## Ensemble

Follow these steps to reproduce the solution:
1. Upload the `Feature_Engineering_&_CATBOOST.ipynb` notebook to colab.
    - Enable GPU runtime
    - Run all to get the catboost_models file

2. Upload the  `Feature_Engineering_&_LGBM.ipynb` notebook to colab.
    - Enable TPU runtime
    - Run all to get the lgbm_models file

3. Upload the `Pixel_Features-Pytorch.ipynb` notebook to colab
    - Enable GPU runtime
    - Run all to get the pytorch_models file

4. Finally upload the `Ensemble.ipynb` notebook to colab
     - Upload the lgbm_models file
     - Upload the catboost_models file
     - Upload the pytorch_models file
     - Run all to get the final submission file
     
Use the below notebooks for data download and manipulation:
- Data Download
- Numpy_Extraction_for_Month_Start_Month_End
- Numpy Extraction_for_25_Periods
- Field_Aggregation_Mean

