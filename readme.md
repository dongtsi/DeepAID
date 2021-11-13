![logo](logo.png)
---
![](https://img.shields.io/badge/license-MIT-green.svg)
![](https://img.shields.io/badge/language-python-blue.svg)
![](https://img.shields.io/badge/framework-pytorch-red.svg)

**DeepAID** is the first *Deep Learning Interpretation* method dedicated for *Anomaly Detection* models in *Security Domains*.  There are three superiorities of DeepAID Interpretations:

- **Unsupervised.** DeepAID is dedicated to interpreting anomaly detection models, which are usually built with only normal data. In DeepAID, not any knowledge of anomaly data is necessary for the interpretation.

- **High Quality.** DeepAID is dedicated to interpreting DL models in security-related domains, where errors are with low tolerance. In DeepAID, the interpretation results are high-quality and satisfies several elegant properties, including fidelity, robustness,  stability, conciseness, and efficiency.   

- **Versatile.** DeepAID not only provides the implementation of certain DL models and anomaly detection systems, but also a general interpretation framework for various types of DL models and security domains.

  

# Implementation Notes

1. Current implementation of DeepAID only supports interpreting DL models built with Pytorch. We'll consider extending DeepAID Interpreter to other DL frameworks such as tensorflow. We also provide [**instructions**](instruction/main.md) for building a customized interpreter if your DL model is not yet supported by our implementation.

2. **Environmental Setup**:
   
   > pip install -r requirement.txt
   
   - For Tabular Interpreter only:
   > pip install -r requirement_tab.txt
   
   - For Univariate Time-Series Interpreter only:
   > pip install -r requirement_units.txt
   
   - For Multivariate Time-Series Interpreter only:
   > pip install -r requirement_multits.txt



# Examples

We provide several cases to show how to interpret your own anomaly detection models, including:

- [Tabular Data, Auto-Encoder, Synthetic Data](demos/tabular_synthesis/tabular_example_synthesis.ipynb)
- [Tabular Data, Kitsune (NDSS'18), Network Intrusion Detection](demos/tabular_kitsune/tabular_example_kitsune.ipynb)
- [Time Series (Univariate), DeepLog (CCS'17), Log Anomaly Detection](demos/timeseries_uni_deeplog/timeseries_example_deeplog.ipynb)
- [Time Series (Multivariate), LSTM, Network Anomaly Detection](demos/timeseries_multi_nids/timeseries_example_nids.ipynb)
- [Graph Data (Link Prediction, Embedding), GL-GV (RAID'20), APT Lateral Movement Detection](demos/graph_link_embed_apt/graph_embed_link_example.ipynb)



# Customizing Interpreters

DeepAID follows a general interpretation framework for various types of DL models and security domains. The core idea of interpreting anomalies in DeepAID is searching a **reference** and interpreting through the **difference** between the **reference** and anomaly. The searching process is limited by several considerations (i.e., constraints) to generate high-qulity results. Here is an illustration:
![framework](instruction/framework.gif)

**See our [paper](https://arxiv.org/abs/2109.11495) for more technical details and the [instruction](instruction/main.md) of building Interpreters for your own models.**



# Citation & Paper 

This source code is part of our work accepted by [CCS'21](https://www.sigsac.org/ccs/CCS2021/):

***DeepAID: Interpreting and Improving Deep Learning-based Anomaly Detection in Security Applications*** 

Its pre-print version is available at [here](https://arxiv.org/abs/2109.11495). 

You can find more details in this paper, and if you use the source code, **please cite the paper**.

(Here is the **BibTex**:)

>```
>@article{DBLP:journals/corr/abs-2109-11495,
>  author    = {Dongqi Han and
>               Zhiliang Wang and
>               Wenqi Chen and
>               Ying Zhong and
>               Su Wang and
>               Han Zhang and
>               Jiahai Yang and
>               Xingang Shi and
>               Xia Yin},
>  title     = {DeepAID: Interpreting and Improving Deep Learning-based Anomaly Detection
>               in Security Applications},
>  journal   = {CoRR},
>  volume    = {abs/2109.11495},
>  year      = {2021}
>}
>```



