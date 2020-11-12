#A Lightweight Network Based on Separable Convolution Using Inverted Residuals For Speech Emotion Recognition
A tensorflow implementation of lightweight model in[]

## Data:
Data discriptions of IEMOCAP please refer to 
## Requirements:
Some required libraaies:
```
python                   >=3.6   
tensorflow-gpu           1.11.0
cuda                      10
```

## code:
main_cr_mel_method1: the main function of the code
cr_model_v2:
    cfgs: config parameters(yaml file)
    cr_model.py: main structure
    cr_model_imple_mel: details about lightweight network
    cr_model_run: train network
    data_set: generator 
    load_data: processes data

scripts: processes dataset, there are some .py file can help you but not well regulated

## How to run
first: preprocess:
            extract_wavs.py
            wav_scripts.py
            wav2log_mel.py
            div_sets.py
            
then: train the network by run main_cr_mel_method1.py

To obtain the best model, we store our model when the accurary increase or the loss decrease

## Citation
Zhong Y, Hu Y, Huang H, et al. A Lightweight Model Based on Separable Convolution for Speech Emotion Recognition[J]. Proc. Interspeech 2020, 2020: 3331-3335.

    
