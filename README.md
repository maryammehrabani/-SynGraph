This code repository includes a novel approach called SynGraph, which leverages the integration of Knowledge Graph representation and HyperGraph modeling to predict the synergistic effects of drug combinations
![image](https://github.com/maryammehrabani/SynKH/assets/93048428/19db0120-f82d-4e92-99c1-20e021cd7376)
 
**Environment Setup**

You can install the packages step by step


 ```
pip install nltk
pip install gensim
pip install dhg
```

**Usage**

You can download all data from  https://github.com/maryammehrabani/SynGraph/tree/master/DATA

Required functions:

```
python function.py
```

**Property Prediction Model**


1.Prepare data

```
python knowledge_graph.py
```
```
python feature_enhancment.py
```
2.Training

```
python main.py
```
