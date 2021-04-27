## BioEE-RL

Codes for the paper "Efficient Multiple Biomedical Events Extraction Via Reinforcement Learning", and you can find our paper at [here](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btab024/6119341).


### Cite
Cite this paper as:
```
@article{10.1093/bioinformatics/btab024,
    author = {Zhao, Weizhong and Zhao, Yao and Jiang, Xingpeng and He, Tingting and Liu, Fan and Li, Ning},
    title = "{Efficient multiple biomedical events extraction via reinforcement learning}",
    journal = {Bioinformatics},
    year = {2021},
    month = {01},
    abstract = "{Multiple events extraction from biomedical literature is a challenging task for biomedical community. Usually, biomedical event extraction is modeled as two sub-tasks, trigger identification and argument detection. Most existing methods perform these two sub-tasks sequentially, and fail to make full use of the interaction between them, leading to suboptimal results for multiple biomedical events extraction.We propose a novel framework of reinforcement learning (RL) for the task of multiple biomedical events extraction. More specifically, trigger identification and argument detection are treated as main-task and subsidiary-task, respectively. Assigning the event type of triggers (in the main-task) is viewed as the action taken in RL, and the result of corresponding argument detection (i.e. the subsidiary-task) for the identified trigger is used for computing the reward of the taken action. Moreover, the result of the subsidiary-task is modeled as part of environment information in RL to help the procedure of trigger identification. In addition, external biomedical knowledge bases are employed for representation learning of biomedical text, which can improve the performance of biomedical event extraction. Results on two widely used biomedical corpora demonstrate that the proposed framework performs better than the selected baselines on the task of multiple events extraction. The ablation test indicates the contributions of RL and external KBs to the performance improvement in the proposed method. In addition, by modeling multiple events extraction under the RL framework, the supervised information is exploited more effectively than the classical supervised learning paradigm.Availability and implementationSource codes will be available at: https://github.com/David-WZhao/BioEE-RL.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab024},
    url = {https://doi.org/10.1093/bioinformatics/btab024},
    note = {btab024},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btab024/36163103/btab024.pdf},
}
```

### Data
- [GE13](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki)
  BioNLP Shared Tasks: Genia event extraction (GE) task, 2013. Originally released by the paper "The Genia Event Extraction Shared Task, 2013 Edition - Overview".
- [MLEE](http://nactem.ac.uk/MLEE/)
  Multi-Level Event Extraction (MLEE) corpus is originally released by the paper "Event extraction across multiple levels of biological organization".

The datasets that we used for event extraction are provided in corresponding subdirectories under BioEE-RL/data.

### Requirements
- Python 3.8.5
- Pytorch 1.6.0