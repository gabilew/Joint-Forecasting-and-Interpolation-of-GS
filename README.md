# Joint Forecasting and Interpolation of Graph Signals Using Deep Learning


![image](https://github.com/gabilew/Spectral-Graph-GRU/blob/master/images/sggru.png)
### Objective: 
* Given a sampled graph signal (i.e.: a low granularity sensor network), interpolate the graph signal to obtain the entire network and make temporal prediction simultaneously.
### Dataset
* Check out this [link](https://github.com/zhiyongc/Seattle-Loop-Data)   for downloading the Seattle loop dataset
* Move the dataset to data/Seattle_Loop_Dataset
### Environment
* Python 3.6.1 and PyTorch 1.4.0
```
conda install pytorch  cudatoolkit=10.0 -c pytorch
```
* Installation: 
```
python setup.py install
```
Reference:

Lewenfus, G., Martins, W. A., Chatzinotas, S., & Ottersten, B. (2020). Joint Forecasting and Interpolation of Graph Signals Using Deep Learning. arXiv preprint arXiv:2006.01536.
