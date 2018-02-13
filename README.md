## [Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation](https://arxiv.org/abs/1406.2984)
#### Maksym Andriushchenko, Fan Yue

The paper became quite influential...

<!-- <img src="images/mnist1-orig.png" width=200/> -->





## Difference from the paper



## How to run the code
1. download FLIC dataset
2. `data.py`
3. `pairwise_distr.py`
4. `python main.py --data_augm --use_sm --optimizer=adam --lr=0.001 --lmbd=0.0001`


Supported options:
- `debug`: 
- `multi_gpu`: BatchNorm can have problems with low batch size
- `data_augm`: 
- `use_sm`: whether to use the Spatial Model (SM) or not
- `restore_model`: if you want to restore an existing model, name of which should be specified in `best_model_name` variable.
Or you can get further information on different arguments with: `python worker.py --help`.


Datasets (you will need to have them in the folder data/, for details please see `data.py`):
- `FLIC`
- `FLIC+`

Note that the script `main.py` saves tensorboard summaries (folder `tb`), model parameters (folder `models_ex`). 


## Contact
For any questions regarding the code please contact Maksym Andriushchenko (m.**my surname**@gmail.com).
Any suggestions are always welcome.

