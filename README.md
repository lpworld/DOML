# DML

This is our implementation for the paper:

**Pan Li and Alexander Tuzhilin. "Dual Metric Learning for Effective and Efficient Cross-Domain Recommendations."  IEEE Transactions on Knowledge and Data Engineering (TKDE). 2021.** [[Paper]](https://ieeexplore.ieee.org/abstract/document/9409658)

**Important:** Due to the confidential agreement with the company, we are not allowed to make the European dataset publicly available. Nevertheless, we provide a sample of the dataset for you to get an understanding of the input strcuture. The Amazon dataset can be accessed at [[here]](http://jmcauley.ucsd.edu/data/amazon/index_2014.html). You are always welcome to use our codes for your own dataset.

We have also include the autoencoder module under the "autoencoder" folder. While we have yet to cleaned up these codes and you might not be able to run them directly, we hope that these codes will be helpful for you tp construct your own autoencoder model for generating user and item embeddings.

**Please cite our TKDE paper if you use our codes. Thanks!** 

Author: Pan Li (https://lpworld.github.io/)

## Environment Settings
We use PyTorch as the backend. 
- PyTorch version:  '1.2.0'

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the parse_args function). 

Run DML:
```
python train.py
```

## Acknowledgement
This implementation is constructed based on our WSDM20 paper [[DDTCDR: Deep Dual Transfer Cross Domain Recommendation]](https://github.com/lpworld/DDTCDR). The authors would also like to thank Vladimir Bobrikov for providing the dataset for evaluation purposes.

Last Update: 2021/06/16
