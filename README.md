# A Pytorch implementation  for “Balanced Open Set Domain Adaptation via Centroid Alignment”



## Dependencies
* **python==3.6.5**
* **pytorch==1.1.0**
* **scipy==1.2.3**
* **numpy==1.18.5**
* **libmr==0.1.9**

## To reproduce the results, please run:
```
python ./demo.py
```

## We also provide a pretrained model. Run the following command to load the model and get the results:
```
python ./demo_only_test.py
```

This work is built on top of S-VAE, as presented in [[1]](#citation)(http://arxiv.org/abs/1804.00891). 

## Bibtex
```
@inproceedings{jing2021balanced,
  title={Balanced open set domain adaptation via centroid alignment},
  author={Jing, Mengmeng and Li, Jingjing and Zhu, Lei and Ding, Zhengming and Lu, Ke and Yang, Yang},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={35},
  number={9},
  pages={8013--8020},
  year={2021}
}
```
