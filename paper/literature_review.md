# Literature Review: Decision Transformer for Emergency Vehicle Corridor Optimization

Compiled 2026-03-17. All citations verified via web search against primary sources (arXiv, ACM DL, IEEE Xplore, NeurIPS/ICML proceedings).

---

## 1. Decision Transformer Papers (2021--2026)

### 1.1 Decision Transformer (Chen et al., 2021)
- **Authors:** Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch
- **Title:** Decision Transformer: Reinforcement Learning via Sequence Modeling
- **Venue:** NeurIPS 2021 (Advances in Neural Information Processing Systems, vol. 34, pp. 15084--15097)
- **Year:** 2021
- **Contribution:** Casts RL as a sequence-modeling problem. A causally masked GPT-style transformer conditions on desired return, past states, and actions to autoregressively emit optimal actions. Matches or exceeds state-of-the-art model-free offline RL on Atari, OpenAI Gym, and Key-to-Door.

### 1.2 Trajectory Transformer (Janner et al., 2021)
- **Authors:** Michael Janner, Qiyang Li, Sergey Levine
- **Title:** Offline Reinforcement Learning as One Big Sequence Modeling Problem
- **Venue:** NeurIPS 2021 (vol. 34, pp. 1273--1286)
- **Year:** 2021
- **Contribution:** Treats states, actions, and rewards as one interleaved token sequence. Uses beam-search-based planning for long-horizon dynamics prediction, imitation, goal-conditioned RL, and offline RL.

### 1.3 Online Decision Transformer (Zheng et al., 2022)
- **Authors:** Qinqing Zheng, Amy Zhang, Aditya Grover
- **Title:** Online Decision Transformer
- **Venue:** ICML 2022 (PMLR 162, pp. 27042--27059)
- **Year:** 2022
- **Contribution:** Unifies offline pretraining with online fine-tuning via sequence-level entropy regularizers. Achieves competitive D4RL performance with significantly faster fine-tuning.

### 1.4 Elastic Decision Transformer (Wu et al., 2023)
- **Authors:** Yueh-Hua Wu, Xiaolong Wang, Masashi Hamaya
- **Title:** Elastic Decision Transformer
- **Venue:** NeurIPS 2023
- **Year:** 2023
- **Contribution:** Addresses DT's inability to perform trajectory stitching by accepting variable-length trajectory histories. Short histories increase exploration; long histories stabilize optimal trajectories. Outperforms offline RL baselines on D4RL and Atari.

### 1.5 Q-learning Decision Transformer (Yamagata et al., 2023)
- **Authors:** Taku Yamagata, Ahmed Khalil, Raul Santos-Rodriguez
- **Title:** Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL
- **Venue:** ICML 2023
- **Year:** 2023
- **Contribution:** Relabels return-to-go targets using Q-learning results before training DT, enabling trajectory stitching from suboptimal data.

### 1.6 Constrained Decision Transformer (Liu et al., 2023)
- **Authors:** Zuxin Liu, Zijian Guo, Yihang Yao, Zhepeng Cen, Wenhao Yu, Tingnan Zhang, Ding Zhao
- **Title:** Constrained Decision Transformer for Offline Safe Reinforcement Learning
- **Venue:** ICML 2023 (pp. 21611--21630)
- **Year:** 2023
- **Contribution:** Formulates offline safe RL as multi-objective optimization. CDT dynamically adjusts safety/performance trade-offs during deployment with zero-shot adaptation to different constraint thresholds.

### 1.7 Offline Pre-trained Multi-Agent Decision Transformer (Meng et al., 2023)
- **Authors:** Linghui Meng, Muning Wen, Chenyang Le, Xiyun Li, Dengpeng Xing, Weinan Zhang, Ying Wen, Haifeng Zhang, Jun Wang, Yaodong Yang, Bo Xu
- **Title:** Offline Pre-trained Multi-Agent Decision Transformer
- **Venue:** Machine Intelligence Research, vol. 20, no. 2, pp. 233--248, 2023
- **Year:** 2023
- **Contribution:** Extends DT to multi-agent settings on StarCraft II. Outperforms BCQ and CQL baselines; pre-trained MADT improves sample efficiency and supports zero-shot/few-shot transfer.

---

## 2. Emergency Vehicle Preemption Papers (2018--2026)

### 2.1 EMVLight -- AAAI version (Su et al., 2022)
- **Authors:** Haoran Su, Yaofeng D. Zhong, Biswadip Dey, Abhishek Chakraborty
- **Title:** EMVLight: A Decentralized Reinforcement Learning Framework for Efficient Passage of Emergency Vehicles
- **Venue:** AAAI 2022 (vol. 36, no. 4, pp. 4593--4601)
- **Year:** 2022
- **Contribution:** Decentralized multi-agent advantage actor-critic with policy sharing for joint EV routing and signal pre-emption. Reduces EV travel time by up to 42.6%.

### 2.2 EMVLight -- Transportation Research Part C version (Su et al., 2023)
- **Authors:** Haoran Su, Yaofeng D. Zhong, Joseph Y.J. Chow, Biswadip Dey, Li Jin
- **Title:** EMVLight: A Multi-Agent Reinforcement Learning Framework for an Emergency Vehicle Decentralized Routing and Traffic Signal Control System
- **Venue:** Transportation Research Part C: Emerging Technologies, vol. 146, 2023
- **Year:** 2023
- **Contribution:** Extended journal version with pressure-based reward and multi-class RL agents. Also reduces non-EV average travel time by 23.5%.

### 2.3 Emergency Vehicle Lane Pre-clearing (Wu et al., 2020)
- **Authors:** Jiawei Wu, Balazs Kulcsar, Soyoung Ahn, Xiaobo Qu
- **Title:** Emergency Vehicle Lane Pre-clearing: From Microscopic Cooperation to Routing Decision Making
- **Venue:** Transportation Research Part B: Methodological, vol. 141, pp. 223--239, 2020
- **Year:** 2020
- **Contribution:** Mixed-integer nonlinear programming formulation for cooperative lane-clearing by connected vehicles. Bi-level optimization balances EV speed guarantee and disturbance minimization.

### 2.4 Hierarchical GNN for EV Corridor Formation (Su et al., 2026)
- **Authors:** Haoran Su et al.
- **Title:** Hierarchical GNN-Based Multi-Agent Learning for Dynamic Queue-Jump Lane and Emergency Vehicle Corridor Formation
- **Venue:** arXiv:2601.04177 (January 2026)
- **Year:** 2026
- **Contribution:** Two-level GNN-MAPPO architecture: high-level global corridor planner + low-level trajectory controllers. Reduces EV travel time by 28.3% with near-zero collision rate (0.3%).

### 2.5 Traffic Signal Priority via Shared Experience MARL (Wang et al., 2023)
- **Authors:** Qiang Wang et al.
- **Title:** Traffic Signal Priority Control Based on Shared Experience Multi-Agent Deep Reinforcement Learning
- **Venue:** IET Intelligent Transport Systems, vol. 17, no. 6, pp. 1150--1163, 2023
- **Year:** 2023
- **Contribution:** Hybrid reward function emphasizing EV priority while maintaining overall traffic efficiency in urban road networks.

### 2.6 EV Preemption Strategies (Qin and Khan, 2012)
- **Authors:** Xiao Qin, Ata M. Khan
- **Title:** Control Strategies for Emergency Vehicle Preemption and Transit Signal Priority
- **Venue:** Journal of Transportation Engineering, vol. 138, no. 1, pp. 93--101, 2012
- **Year:** 2012
- **Contribution:** Classic review of conventional signal preemption approaches (early return to green, phase insertion). Establishes traditional baseline methods.

---

## 3. Multi-Agent RL for Traffic Signal Control (2019--2026)

### 3.1 PressLight (Wei et al., 2019)
- **Authors:** Hua Wei, Chacha Chen, Guanjie Zheng, Kan Wu, Vikash Gayah, Kai Xu, Zhenhui Li
- **Title:** PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network
- **Venue:** KDD 2019 (pp. 1290--1298)
- **Year:** 2019
- **Contribution:** Combines max pressure theory with RL. Reward design proven to maximize network throughput. Outperforms both conventional and RL baselines.

### 3.2 CoLight (Wei et al., 2019)
- **Authors:** Hua Wei, Nan Xu, Huichu Zhang, Guanjie Zheng, Xinshi Zang, Chacha Chen, Weinan Zhang, Yanmin Zhu, Kai Xu, Zhenhui Li
- **Title:** CoLight: Learning Network-Level Cooperation for Traffic Signal Control
- **Venue:** CIKM 2019 (pp. 1913--1922)
- **Year:** 2019
- **Contribution:** First to use graph attention networks for RL-based TSC. Models spatial and temporal influences of neighboring intersections. Scales to 196-intersection real-world networks.

### 3.3 MPLight (Chen et al., 2020)
- **Authors:** Chacha Chen, Hua Wei, Nan Xu, Guanjie Zheng, Ming Yang, Yuanhao Xiong, Kai Xu, Zhenhui Li
- **Title:** Toward a Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control
- **Venue:** AAAI 2020 (vol. 34, no. 04, pp. 3414--3421)
- **Year:** 2020
- **Contribution:** Decentralized RL with parameter sharing for city-scale TSC. Successfully controls 2510 intersections in Manhattan via pressure-based reward for implicit coordination.

### 3.4 FRAP (Zheng et al., 2019)
- **Authors:** Guanjie Zheng, Yuanhao Xiong, Xinshi Zang, Jie Feng, Hua Wei, Huichu Zhang, Yong Li, Kai Xu, Zhenhui Li
- **Title:** Learning Phase Competition for Traffic Signal Control
- **Venue:** CIKM 2019 (pp. 1963--1972)
- **Year:** 2019
- **Contribution:** Models phase competition: conflicting signals prioritize higher-demand movements. Achieves invariance to symmetric cases (flipping/rotation), faster convergence, superior generalizability.

### 3.5 Large-Scale MARL for TSC (Chu et al., 2019)
- **Authors:** Tianshu Chu, Jie Wang, Lara Codeca, Zhaojian Li
- **Title:** Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control
- **Venue:** IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 3, pp. 1086--1095, 2019
- **Year:** 2019
- **Contribution:** Scalable multi-agent DRL framework with independent DQN agents for large-scale TSC. Demonstrates feasibility of decentralized learning at scale.

---

## 4. Transformer-based Traffic Signal Control (2023--2026)

### 4.1 TransformerLight (Wu et al., 2023)
- **Authors:** Qiang Wu, Mingyuan Li, Jun Shen, Linyuan Lu, Bo Du, Ke Zhang
- **Title:** TransformerLight: A Novel Sequence Modeling Based Traffic Signaling Mechanism via Gated Transformer
- **Venue:** KDD 2023
- **Year:** 2023
- **Contribution:** Formulates TSC as sequence modeling. Replaces residual connections with gated transformer blocks for training stability. Trained on 20% historical data; transfers to other real-world datasets.

### 4.2 DTLight (Huang et al., 2023)
- **Authors:** Xingshuai Huang, Di Wu, Benoit Boulet
- **Title:** Traffic Signal Control Using Lightweight Transformers: An Offline-to-Online RL Approach
- **Venue:** arXiv:2312.07795 (December 2023)
- **Year:** 2023
- **Contribution:** First offline-to-online DT approach for single- and multi-intersection TSC. Introduces DTRL (16 offline TSC datasets). Uses knowledge distillation for lightweight deployment and adapter modules for online fine-tuning.

### 4.3 X-Light (Jiang et al., 2024)
- **Authors:** Haoyuan Jiang, Ziyue Li, Hua Wei, Xuantang Xiong, Jingqing Ruan, Jiaming Lu, Hangyu Mao, Rui Zhao
- **Title:** X-Light: Cross-City Traffic Signal Control Using Transformer on Transformer as Meta Multi-Agent Reinforcement Learner
- **Venue:** IJCAI 2024
- **Year:** 2024
- **Contribution:** Dual-level Transformer on Transformer (TonT): Lower Transformer aggregates intersection-level MDP info, Upper Transformer learns cross-city decision trajectories. Surpasses baselines by +7.91% (up to +16.3%) on zero-shot cross-city transfer.

### 4.4 Sequence Decision Transformer (Zhao et al., 2024)
- **Authors:** Rui Zhao, Haofeng Hu et al.
- **Title:** Sequence Decision Transformer for Adaptive Traffic Signal Control
- **Venue:** Sensors, vol. 24, no. 19, article 6202, 2024
- **Year:** 2024
- **Contribution:** Transformer-based actor-critic architecture for ATSC modeled as MDP. Combines DRL attention mechanisms with sequence decision capabilities for urban traffic management.

### 4.5 Spatiotemporal Decision Transformer (Su et al., 2026)
- **Authors:** Haoran Su et al.
- **Title:** Spatiotemporal Decision Transformer for Traffic Coordination
- **Venue:** arXiv:2602.02903 (February 2026)
- **Year:** 2026
- **Contribution:** Extends DT to multi-agent TSC with (1) graph attention for spatial dependencies, (2) temporal transformer encoder, (3) return-to-go conditioning. Reduces average travel time by 5--6% vs. strongest baselines.

---

## 5. Offline RL (2019--2024)

### 5.1 CQL (Kumar et al., 2020)
- **Authors:** Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine
- **Title:** Conservative Q-Learning for Offline Reinforcement Learning
- **Venue:** NeurIPS 2020 (vol. 33, pp. 1179--1191)
- **Year:** 2020
- **Contribution:** Learns conservative Q-function providing lower bound on true policy value. Augments Bellman error with Q-value regularizer. Attains 2--5x higher returns than prior offline RL on complex multi-modal data.

### 5.2 BCQ (Fujimoto et al., 2019)
- **Authors:** Scott Fujimoto, David Meger, Doina Precup
- **Title:** Off-Policy Deep Reinforcement Learning without Exploration
- **Venue:** ICML 2019 (pp. 2052--2062)
- **Year:** 2019
- **Contribution:** First batch/offline deep RL algorithm. Constrains action space via a generative model to keep policy close to data distribution. Uses VAE + perturbation model.

### 5.3 IQL (Kostrikov et al., 2022)
- **Authors:** Ilya Kostrikov, Ashvin Nair, Sergey Levine
- **Title:** Offline Reinforcement Learning with Implicit Q-Learning
- **Venue:** ICLR 2022
- **Year:** 2022
- **Contribution:** Never evaluates out-of-distribution actions. Uses expectile regression on value function + advantage-weighted behavioral cloning. State-of-the-art on D4RL.

### 5.4 TD3+BC (Fujimoto and Gu, 2021)
- **Authors:** Scott Fujimoto, Shixiang Shane Gu
- **Title:** A Minimalist Approach to Offline Reinforcement Learning
- **Venue:** NeurIPS 2021 (vol. 34, pp. 20132--20145)
- **Year:** 2021
- **Contribution:** Simply adds behavior cloning loss to TD3 policy update + state normalization. Matches state-of-the-art on D4RL with minimal implementation complexity. De-facto offline RL baseline.

### 5.5 Offline RL Tutorial (Levine et al., 2020)
- **Authors:** Sergey Levine, Aviral Kumar, George Tucker, Justin Fu
- **Title:** Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems
- **Venue:** arXiv:2005.01643, 2020
- **Year:** 2020
- **Contribution:** Comprehensive tutorial covering offline RL formulation, key challenges (distributional shift, extrapolation error), algorithmic families, and open problems.

### 5.6 Offline RL Survey (Prudencio et al., 2023)
- **Authors:** Rafael Figueiredo Prudencio, Marcos R.O.A. Maximo, Esther Luna Colombini
- **Title:** A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems
- **Venue:** IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 8, pp. 10237--10257, 2023
- **Year:** 2023
- **Contribution:** Unifying taxonomy for offline RL methods with comprehensive review using unified notation. Covers benchmarks and open problems.

---

## 6. Offline RL for Traffic

### 6.1 DataLight (Zhang et al., 2023)
- **Authors:** Liang Zhang, Yutong Zhang, Jianming Deng, Chen Li
- **Title:** DataLight: Offline Data-Driven Traffic Signal Control
- **Venue:** arXiv:2303.10828, 2023
- **Year:** 2023
- **Contribution:** Offline RL approach for TSC that learns entirely from logged data without environment interaction. Demonstrates feasibility of pure data-driven signal control.

---

## 7. Cell Transmission Model / Traffic Simulation

### 7.1 CTM Part I (Daganzo, 1994)
- **Authors:** Carlos F. Daganzo
- **Title:** The Cell Transmission Model: A Dynamic Representation of Highway Traffic Consistent with the Hydrodynamic Theory
- **Venue:** Transportation Research Part B: Methodological, vol. 28, no. 4, pp. 269--287, 1994
- **Year:** 1994
- **Contribution:** Foundational macroscopic traffic model. Partitions highway into cells; updates flow via send/receive comparison consistent with kinematic wave theory. Automatically generates shockwaves.

### 7.2 CTM Part II (Daganzo, 1995)
- **Authors:** Carlos F. Daganzo
- **Title:** The Cell Transmission Model, Part II: Network Traffic
- **Venue:** Transportation Research Part B: Methodological, vol. 29, no. 2, pp. 79--93, 1995
- **Year:** 1995
- **Contribution:** Extends CTM to multi-commodity flows on complex networks with junctions, merges, and diverges.

### 7.3 CityFlow (Zhang et al., 2019)
- **Authors:** Huichu Zhang, Siyuan Feng, Chang Liu, Yaoyao Ding, Yichen Zhu, Zihan Zhou, Weinan Zhang, Yong Yu, Haiming Jin, Zhenhui Li
- **Title:** CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario
- **Venue:** WWW 2019 (pp. 3620--3624)
- **Year:** 2019
- **Contribution:** 20x faster than SUMO. Supports flexible road network definitions, synthetic/real-world data, user-friendly RL interface with multithreaded acceleration.

### 7.4 SUMO (Lopez et al., 2018)
- **Authors:** Pablo Alvarez Lopez, Michael Behrisch, Laura Bieker-Walz, Jakob Erdmann, Yun-Pang Flotterod, Robert Hilbrich, Leonhard Lucken, Johannes Rummel, Peter Wagner, Evamarie Wiessner
- **Title:** Microscopic Traffic Simulation using SUMO
- **Venue:** IEEE ITSC 2018 (pp. 2575--2582)
- **Year:** 2018
- **Contribution:** Open-source microscopic traffic simulation. Space-continuous, time-discrete car-following model. De-facto standard for traffic research with extensive tool ecosystem.

---

## 8. Foundation Architecture

### 8.1 Transformer (Vaswani et al., 2017)
- **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- **Title:** Attention is All You Need
- **Venue:** NeurIPS 2017 (vol. 30)
- **Year:** 2017
- **Contribution:** Introduces the Transformer architecture based solely on self-attention. Enables parallelizable training and achieves SOTA on machine translation. Foundation for all DT work.

### 8.2 Graph Attention Networks (Velickovic et al., 2018)
- **Authors:** Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, Yoshua Bengio
- **Title:** Graph Attention Networks
- **Venue:** ICLR 2018
- **Year:** 2018
- **Contribution:** Introduces attention-based message passing on graphs. Foundation for graph-based multi-agent communication in TSC methods like CoLight.

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Decision Transformer variants | 7 |
| Emergency Vehicle Preemption | 6 |
| Multi-Agent RL for TSC | 5 |
| Transformer-based TSC | 5 |
| Offline RL algorithms | 6 |
| Offline RL for Traffic | 1 |
| CTM / Simulation | 4 |
| Foundation (Transformer, GAT) | 2 |
| **Total** | **36** |
