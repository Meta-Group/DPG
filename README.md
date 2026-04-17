# Decision Predicate Graph (DPG)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](pyproject.toml)
[![Build Status](https://github.com/Meta-Group/DPG/actions/workflows/ci.yml/badge.svg)](https://github.com/Meta-Group/DPG/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/dpg/badge/?version=latest)](https://dpg.readthedocs.io/en/latest/)

<p align="center">
  <img src="https://github.com/Meta-Group/DPG/blob/main/DPG.png" width="300" />
</p>


DPG is a model-agnostic tool to provide a global interpretation of tree-based ensemble models, addressing transparency and explainability challenges.

DPG is a graph structure that captures the tree-based ensemble model and learned dataset details, preserving the relations among features, logical decisions, and predictions towards emphasising insightful points.
DPG enables graph-based evaluations and the identification of model decisions towards facilitating comparisons between features and their associated values while offering insights into the entire model.
DPG provides descriptive metrics that enhance the understanding of the decisions inherent in the model, offering valuable insights.

```mermaid
flowchart LR
    979618565717729770316449045241379584267506428030["F1 &lt;= 0.15"]
    style 979618565717729770316449045241379584267506428030 fill:#deebf7,color:#000
    400374097622164785150640648135130385802573046609["F4 &gt; 8.595"]
    style 400374097622164785150640648135130385802573046609 fill:#deebf7,color:#000
    588807308583655218865209145046442769682314462588["Class C"]
    style 588807308583655218865209145046442769682314462588 fill:#9dc3e6,color:#000
    100415085763263903852039295602911702948457746142["F4 &lt;= 10.305"]
    style 100415085763263903852039295602911702948457746142 fill:#deebf7,color:#000
    796167064376282459648852036451958843494915092088["F4 &lt;= 9.23"]
    style 796167064376282459648852036451958843494915092088 fill:#deebf7,color:#000
    1210267766209222920336595164156346343579215270195["Class D"]
    style 1210267766209222920336595164156346343579215270195 fill:#9dc3e6,color:#000
    1453090674414881797846805718821638125042633934677["F4 &gt; 9.23"]
    style 1453090674414881797846805718821638125042633934677 fill:#deebf7,color:#000
    1018563747240538230561512835449441483670234791480["F1 &lt;= 0.275"]
    style 1018563747240538230561512835449441483670234791480 fill:#deebf7,color:#000
    604794083049618193149218731106783663700431359465["F6 &lt;= 36.634998"]
    style 604794083049618193149218731106783663700431359465 fill:#deebf7,color:#000
    219056158314876120791781277892726213829941686289["F1 &gt; 0.275"]
    style 219056158314876120791781277892726213829941686289 fill:#deebf7,color:#000
    971674044816125407632765233907262716905523475854["F4 &lt;= 8.595"]
    style 971674044816125407632765233907262716905523475854 fill:#deebf7,color:#000
    690715845280696214418205697557183424080425915385["Class B"]
    style 690715845280696214418205697557183424080425915385 fill:#9dc3e6,color:#000
    373114052074461482953194598311845305553485338248["F3 &lt;= 0.43"]
    style 373114052074461482953194598311845305553485338248 fill:#deebf7,color:#000
    130059101401781709675883937374914161454065576473["F7 &gt; -1.54"]
    style 130059101401781709675883937374914161454065576473 fill:#deebf7,color:#000
    946778044485377185187951514693838972119007272915["F3 &lt;= 0.515"]
    style 946778044485377185187951514693838972119007272915 fill:#deebf7,color:#000
    931261134179154107191046451665025768645110437780["F7 &lt;= -5.635"]
    style 931261134179154107191046451665025768645110437780 fill:#deebf7,color:#000
    581835747318489062599131910623234493662676984432["Class A"]
    style 581835747318489062599131910623234493662676984432 fill:#9dc3e6,color:#000
    95328426820969179252858136277969691480959498483["F7 &gt; -6.97"]
    style 95328426820969179252858136277969691480959498483 fill:#deebf7,color:#000
    838025964173435133002132899454873804629558551595["F1 &lt;= 0.265"]
    style 838025964173435133002132899454873804629558551595 fill:#deebf7,color:#000
    157558602123247801017580030895190959370148853051["F7 &gt; -5.635"]
    style 157558602123247801017580030895190959370148853051 fill:#deebf7,color:#000
    15000782385466634000392697558394770189706855891["F10 &lt;= 91.349998"]
    style 15000782385466634000392697558394770189706855891 fill:#deebf7,color:#000
    737571175891001486850004482015570617772703609595["F7 &gt; 4.94"]
    style 737571175891001486850004482015570617772703609595 fill:#deebf7,color:#000
    820268771082212553636570576938320986082031670022["F7 &lt;= 4.755"]
    style 820268771082212553636570576938320986082031670022 fill:#deebf7,color:#000
    222225252368107721717009608605956582502482651473["F10 &gt; 91.349998"]
    style 222225252368107721717009608605956582502482651473 fill:#deebf7,color:#000
    1290399266530111018100832136017555924837222381379["F6 &gt; 38.915001"]
    style 1290399266530111018100832136017555924837222381379 fill:#deebf7,color:#000
    1432524820656786198276600212795640928391034583565["F9 &lt;= 32.765001"]
    style 1432524820656786198276600212795640928391034583565 fill:#deebf7,color:#000
    1216879825140622668422253624475828723202673476705["F3 &gt; 0.515"]
    style 1216879825140622668422253624475828723202673476705 fill:#deebf7,color:#000
    721812667918485594407398948094746476301162165281["F6 &gt; 36.634998"]
    style 721812667918485594407398948094746476301162165281 fill:#deebf7,color:#000
    320172993420859182112091033633361489749254219598["F1 &gt; 0.265"]
    style 320172993420859182112091033633361489749254219598 fill:#deebf7,color:#000
    1376178018489663700667465426715142260552127285659["F7 &lt;= -1.54"]
    style 1376178018489663700667465426715142260552127285659 fill:#deebf7,color:#000
    1425576715341752496625268860315092560916679831871["F10 &lt;= 100.09"]
    style 1425576715341752496625268860315092560916679831871 fill:#deebf7,color:#000
    1187091628211292434711457323232000634217732658464["F3 &gt; 0.43"]
    style 1187091628211292434711457323232000634217732658464 fill:#deebf7,color:#000
    464506691893636266372305649238338629786578496086["F6 &lt;= 38.915001"]
    style 464506691893636266372305649238338629786578496086 fill:#deebf7,color:#000
    1410694826222849576728516717748878457342387233404["F2 &lt;= 3.075"]
    style 1410694826222849576728516717748878457342387233404 fill:#deebf7,color:#000
    50305701467499117808883646224307030877008964123["F1 &gt; 0.15"]
    style 50305701467499117808883646224307030877008964123 fill:#deebf7,color:#000
    752639284302739662244140925771936984286998982681["F4 &gt; 10.305"]
    style 752639284302739662244140925771936984286998982681 fill:#deebf7,color:#000
    421447506865222128999782636005206948808242116589["F2 &gt; 3.075"]
    style 421447506865222128999782636005206948808242116589 fill:#deebf7,color:#000
    517945134451617424034782581737551822021612382097["F7 &gt; 4.755"]
    style 517945134451617424034782581737551822021612382097 fill:#deebf7,color:#000
    732610326773121789369223421019172016167699952227["F2 &lt;= 3.085"]
    style 732610326773121789369223421019172016167699952227 fill:#deebf7,color:#000
    75153629346265600498761343753901957591568438946["F7 &lt;= -6.97"]
    style 75153629346265600498761343753901957591568438946 fill:#deebf7,color:#000
    1309043412683735018336339302863613748313349494003["F2 &gt; 3.085"]
    style 1309043412683735018336339302863613748313349494003 fill:#deebf7,color:#000
    1137628253617425639228459882552895355870584674508["F7 &lt;= 4.94"]
    style 1137628253617425639228459882552895355870584674508 fill:#deebf7,color:#000
    1220756731626743842054238342928376272838209981252["F10 &gt; 100.09"]
    style 1220756731626743842054238342928376272838209981252 fill:#deebf7,color:#000
    466099670086209313380749716948554630644411676678["F9 &gt; 32.765001"]
    style 466099670086209313380749716948554630644411676678 fill:#deebf7,color:#000
    979618565717729770316449045241379584267506428030 -->|"1.0000"| 400374097622164785150640648135130385802573046609
    979618565717729770316449045241379584267506428030 -->|"3.0000"| 971674044816125407632765233907262716905523475854
    400374097622164785150640648135130385802573046609 -->|"1.0000"| 588807308583655218865209145046442769682314462588
    100415085763263903852039295602911702948457746142 -->|"1.0000"| 796167064376282459648852036451958843494915092088
    100415085763263903852039295602911702948457746142 -->|"2.0000"| 1453090674414881797846805718821638125042633934677
    796167064376282459648852036451958843494915092088 -->|"1.0000"| 1210267766209222920336595164156346343579215270195
    1453090674414881797846805718821638125042633934677 -->|"2.0000"| 588807308583655218865209145046442769682314462588
    1018563747240538230561512835449441483670234791480 -->|"3.0000"| 604794083049618193149218731106783663700431359465
    1018563747240538230561512835449441483670234791480 -->|"7.0000"| 721812667918485594407398948094746476301162165281
    604794083049618193149218731106783663700431359465 -->|"3.0000"| 588807308583655218865209145046442769682314462588
    219056158314876120791781277892726213829941686289 -->|"3.0000"| 100415085763263903852039295602911702948457746142
    219056158314876120791781277892726213829941686289 -->|"15.0000"| 752639284302739662244140925771936984286998982681
    971674044816125407632765233907262716905523475854 -->|"3.0000"| 690715845280696214418205697557183424080425915385
    373114052074461482953194598311845305553485338248 -->|"3.0000"| 130059101401781709675883937374914161454065576473
    373114052074461482953194598311845305553485338248 -->|"9.0000"| 1376178018489663700667465426715142260552127285659
    130059101401781709675883937374914161454065576473 -->|"3.0000"| 588807308583655218865209145046442769682314462588
    946778044485377185187951514693838972119007272915 -->|"4.0000"| 931261134179154107191046451665025768645110437780
    946778044485377185187951514693838972119007272915 -->|"18.0000"| 157558602123247801017580030895190959370148853051
    931261134179154107191046451665025768645110437780 -->|"4.0000"| 581835747318489062599131910623234493662676984432
    95328426820969179252858136277969691480959498483 -->|"4.0000"| 838025964173435133002132899454873804629558551595
    95328426820969179252858136277969691480959498483 -->|"9.0000"| 320172993420859182112091033633361489749254219598
    838025964173435133002132899454873804629558551595 -->|"4.0000"| 581835747318489062599131910623234493662676984432
    157558602123247801017580030895190959370148853051 -->|"4.0000"| 979618565717729770316449045241379584267506428030
    157558602123247801017580030895190959370148853051 -->|"14.0000"| 50305701467499117808883646224307030877008964123
    15000782385466634000392697558394770189706855891 -->|"5.0000"| 737571175891001486850004482015570617772703609595
    15000782385466634000392697558394770189706855891 -->|"29.0000"| 1137628253617425639228459882552895355870584674508
    15000782385466634000392697558394770189706855891 -->|"35.0000"| 588807308583655218865209145046442769682314462588
    737571175891001486850004482015570617772703609595 -->|"5.0000"| 1210267766209222920336595164156346343579215270195
    820268771082212553636570576938320986082031670022 -->|"5.0000"| 222225252368107721717009608605956582502482651473
    820268771082212553636570576938320986082031670022 -->|"35.0000"| 15000782385466634000392697558394770189706855891
    222225252368107721717009608605956582502482651473 -->|"10.0000"| 690715845280696214418205697557183424080425915385
    1290399266530111018100832136017555924837222381379 -->|"5.0000"| 222225252368107721717009608605956582502482651473
    1290399266530111018100832136017555924837222381379 -->|"34.0000"| 15000782385466634000392697558394770189706855891
    1432524820656786198276600212795640928391034583565 -->|"5.0000"| 1216879825140622668422253624475828723202673476705
    1432524820656786198276600212795640928391034583565 -->|"22.0000"| 946778044485377185187951514693838972119007272915
    1216879825140622668422253624475828723202673476705 -->|"5.0000"| 581835747318489062599131910623234493662676984432
    721812667918485594407398948094746476301162165281 -->|"7.0000"| 690715845280696214418205697557183424080425915385
    320172993420859182112091033633361489749254219598 -->|"9.0000"| 690715845280696214418205697557183424080425915385
    1376178018489663700667465426715142260552127285659 -->|"9.0000"| 690715845280696214418205697557183424080425915385
    1425576715341752496625268860315092560916679831871 -->|"10.0000"| 1187091628211292434711457323232000634217732658464
    1425576715341752496625268860315092560916679831871 -->|"12.0000"| 373114052074461482953194598311845305553485338248
    1187091628211292434711457323232000634217732658464 -->|"10.0000"| 690715845280696214418205697557183424080425915385
    464506691893636266372305649238338629786578496086 -->|"10.0000"| 1018563747240538230561512835449441483670234791480
    464506691893636266372305649238338629786578496086 -->|"18.0000"| 219056158314876120791781277892726213829941686289
    1410694826222849576728516717748878457342387233404 -->|"13.0000"| 95328426820969179252858136277969691480959498483
    1410694826222849576728516717748878457342387233404 -->|"24.0000"| 75153629346265600498761343753901957591568438946
    50305701467499117808883646224307030877008964123 -->|"14.0000"| 690715845280696214418205697557183424080425915385
    752639284302739662244140925771936984286998982681 -->|"15.0000"| 1210267766209222920336595164156346343579215270195
    421447506865222128999782636005206948808242116589 -->|"19.0000"| 517945134451617424034782581737551822021612382097
    421447506865222128999782636005206948808242116589 -->|"40.0000"| 820268771082212553636570576938320986082031670022
    517945134451617424034782581737551822021612382097 -->|"19.0000"| 1210267766209222920336595164156346343579215270195
    732610326773121789369223421019172016167699952227 -->|"22.0000"| 1425576715341752496625268860315092560916679831871
    732610326773121789369223421019172016167699952227 -->|"34.0000"| 1220756731626743842054238342928376272838209981252
    75153629346265600498761343753901957591568438946 -->|"24.0000"| 581835747318489062599131910623234493662676984432
    1309043412683735018336339302863613748313349494003 -->|"28.0000"| 464506691893636266372305649238338629786578496086
    1309043412683735018336339302863613748313349494003 -->|"39.0000"| 1290399266530111018100832136017555924837222381379
    1137628253617425639228459882552895355870584674508 -->|"29.0000"| 588807308583655218865209145046442769682314462588
    1220756731626743842054238342928376272838209981252 -->|"34.0000"| 581835747318489062599131910623234493662676984432
    466099670086209313380749716948554630644411676678 -->|"37.0000"| 1410694826222849576728516717748878457342387233404
    466099670086209313380749716948554630644411676678 -->|"59.0000"| 421447506865222128999782636005206948808242116589
```

---

## The structure
The concept behind DPG is to convert a generic tree-based ensemble model for classification into a graph, where:
- Nodes represent predicates, i.e., the feature-value associations present in each node of every tree;
- Edges denote the frequency with which these predicates are satisfied during the model training phase by the samples of the dataset.

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 70, "rankSpacing": 15}}}%%

flowchart TB
    subgraph trees ["Tree Base Learners"]
        direction TB
        subgraph tree2 [" "]
            direction TB
            T2R(("F1, val3"))
            T2C["Class"]
            T2L(("F2, val2"))
            T2RL["..."]
            T2RR["..."]
            T2R -- "≤" --> T2C
            T2R -- ">" --> T2L
            T2L -- "≤" --> T2RL
            T2L -- ">" --> T2RR
        end
        subgraph tree1 [" "]
            direction TB
            T1R(("F1, val1"))
            T1L(("F2, val2"))
            T1C["Class"]
            T1LL["..."]
            T1LR["..."]
            T1R -- "≤" --> T1L
            T1R -- ">" --> T1C
            T1L -- "≤" --> T1LL
            T1L -- ">" --> T1LR
        end
    end

    trees ==> dpg

    subgraph dpg ["DPG"]
        direction TB
        
        E["F2 ≤ val2"]
        FF["F2 > val2"]
        G["Class"]
        EoutLL["..."]
        EoutLR["..."]

        E -- "w4" --> EoutLL
        FF -- "w5" --> EoutLR

        subgraph f1line [" "]
            direction LR
            A["F1 ≤ val1"]
            B["F1 > val3"]
            C["F1 > val1"]
            D["F1 ≤ val3"]
        end

        A -- "w1" --> E
        A -- "w6" --> FF
        B -- "w2" --> E
        B -- "w7" --> FF
        C -- "w3" --> G
        D -- "w8" --> G
    end

    style f1line fill:transparent,stroke:transparent,color:transparent
    style T1C fill:#a7d294,color:#000,stroke:#5a8a4a
    style T2C fill:#a7d294,color:#000,stroke:#5a8a4a
    style T1R fill:#e1f0db,color:#000,stroke:#89a
    style T1L fill:#e1f0db,color:#000,stroke:#89a
    style T2R fill:#e1f0db,color:#000,stroke:#89a
    style T2L fill:#e1f0db,color:#000,stroke:#89a
    style G fill:#4a86c8,color:#fff,stroke:#336
    style A fill:#d4e4f7,color:#000,stroke:#89a
    style B fill:#d4e4f7,color:#000,stroke:#89a
    style C fill:#d4e4f7,color:#000,stroke:#89a
    style D fill:#d4e4f7,color:#000,stroke:#89a
    style E fill:#d4e4f7,color:#000,stroke:#89a
    style FF fill:#d4e4f7,color:#000,stroke:#89a
    style T1LL fill:transparent,stroke:transparent
    style T1LR fill:transparent,stroke:transparent
    style T2RL fill:transparent,stroke:transparent
    style T2RR fill:transparent,stroke:transparent
    style EoutLL fill:transparent,stroke:transparent
    style EoutLR fill:transparent,stroke:transparent
    style EoutLR fill:transparent,stroke:transparent
    style tree2 fill:transparent,stroke:transparent
    style tree1 fill:transparent,stroke:transparent

```

## Metrics
The graph-based nature of DPG provides significant enhancements in the direction of a complete mapping of the ensemble structure.
| Property     | Definition | Utility |
|--------------|------------|---------|
| _Constraints_  | The intervals of values for each feature obtained from all predicates connected by a path that culminates in a given class. | Calculate the classification boundary values of each feature associated with each class. |
| _Betweenness centrality_ | Quantifies the fraction of all the shortest paths between every pair of nodes of the graph passing through the considered node. | Identify potential bottleneck nodes that correspond to crucial decisions. |
| _Local reaching centrality_ | Quantifies the proportion of other nodes reachable from the local node through its outgoing edges. | Assess the importance of nodes similarly to feature importance, but enrich the information by encompassing the values associated with features across all decisions. |
| _Community_ | A subset of nodes of the DPG which is characterised by dense interconnections between its elements and sparse connections with the other nodes of the DPG that do not belong to the community. | Understanding the characteristics of nodes to be assigned to a particular community class, identifying predominant predicates, and those that play a marginal role in the classification process. |


|Constraints | Betweenness centrality | Local reaching centrality | Community|
|------------|------------|--------------|--------------------|
![](https://github.com/Meta-Group/DPG/blob/main/dpg_image_examples/example_constraints.png) | ![](https://github.com/Meta-Group/DPG/blob/main/dpg_image_examples/example_bc.png) | ![](https://github.com/Meta-Group/DPG/blob/main/dpg_image_examples/example_lrc.png) | ![](https://github.com/Meta-Group/DPG/blob/main/dpg_image_examples/example_community.png) |
|Constraints(Class 1) = val3 < F1 ≤ val1, F2 ≤ val2 | BC(F2 ≤ val2) = 4/24 | LRC(F1 ≤ val1) = 6 / 7 | Community(Class 1) = F1 ≤ val1, F2 ≤ val2 |

---
## Installation

To install DPG locally, first clone the repository:

```bash
git clone https://github.com/Meta-Group/DPG.git
cd DPG
```

Then, install the DPG library in development mode using `pip`:
```bash
pip install -e .  
```

Alternatively, if using `pip directly`:
```bash
pip install git+https://github.com/Meta-Group/DPG.git
```
**Troubleshooting:** If you encounter dependency conflicts, we recommend using a virtual environment:

1- For Windows Users:
  ```bash
  # Create a virtual environment
  python -m venv .venv

  # Activate the virtual environment
  .venv\Scripts\activate

  # If you get execution policy errors, run this first in PowerShell as Administrator:
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

  # Then install DPG
  pip install -r ./requirements.txt
  ```
2- For Linux/Mac Users:
  ```bash
  # Create a virtual environment
  python -m venv .venv

  # Activate the virtual environment
  source .venv/bin/activate

  # Install DPG
  pip install -r ./requirements.txt
  ```
3- Deactivating the Virtual Environment:
  When you're done working with DPG, you can deactivate the virtual environment:
  ```bash
  deactivate
  ```

4- Graph rendering error (`dot` not found):
  DPG plotting requires the Graphviz system executable (`dot`) in your PATH.  
  Installing the Python package `graphviz` is not sufficient on its own.

  - macOS (Homebrew):
    ```bash
    brew install graphviz
    ```
  - Ubuntu/Debian:
    ```bash
    sudo apt-get install graphviz
    ```
  - Windows (winget):
    ```powershell
    winget install Graphviz.Graphviz
    ```
---

## Documentation

For full documentation, visit [https://dpg.readthedocs.io/](https://dpg.readthedocs.io/).

To build and serve documentation locally, see [docs/README.md](docs/README.md).

Practitioner-focused benchmark material for `perc_var` and `decimal_threshold` is available in
[tutorials/parameter_sensitivity_benchmark/README.md](tutorials/parameter_sensitivity_benchmark/README.md).

---

## Example usage (Python)

You can also try DPG directly inside a Jupyter Notebook. Here's a minimal working example using the high-level API:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dpg import DPGExplainer

# Load dataset (last column assumed to be target)
df = pd.read_csv("datasets/custom.csv", index_col=0)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train a simple Random Forest classifier
model = RandomForestClassifier(n_estimators=10, random_state=27)
model.fit(X, y)

# Build the DPG and extract global explanations
explainer = DPGExplainer(
    model=model,
    feature_names=X.columns,
    target_names=np.unique(y).astype(str).tolist(),
)
explanation = explainer.explain_global(X.values, communities=True)

# Render the graph to disk
explainer.plot("dpg_output", explanation, save_dir="datasets", export_pdf=True)
explainer.plot_communities("dpg_output", explanation, save_dir="datasets", export_pdf=True)
```

### Legacy API (low-level)

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dpg.core import DecisionPredicateGraph
from dpg.visualizer import plot_dpg
from metrics.nodes import NodeMetrics
from metrics.edges import EdgeMetrics

df = pd.read_csv("datasets/custom.csv", index_col=0)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = RandomForestClassifier(n_estimators=10, random_state=27)
model.fit(X, y)

feature_names = X.columns.tolist()
class_names = np.unique(y).astype(str).tolist()
dpg = DecisionPredicateGraph(
    model=model,
    feature_names=feature_names,
    target_names=class_names
)
dot = dpg.fit(X.values)
dpg_model, nodes_list = dpg.to_networkx(dot)

df_edges = EdgeMetrics.extract_edge_metrics(dpg_model, nodes_list)
df_nodes = NodeMetrics.extract_node_metrics(dpg_model, nodes_list)

plot_dpg(
    "dpg_output",
    dot,
    df_nodes,
    df_edges,
    save_dir="datasets",
    class_flag=True,
    export_pdf=True,
)
```
#### Output:
<p align="center">
  <img src="https://github.com/Meta-Group/DPG/blob/main/dpg_image_examples/dpg_output_communities.png?raw=true" width="600" />
</p>

### API overview (high-level)

The high-level API is designed to return structured outputs so downstream tools can use them directly.

- `DPGExplainer.fit(X)`: builds the DPG structure
- `DPGExplainer.explain_global(X=None, communities=False, community_threshold=0.2)`: returns a `DPGExplanation`
- `DPGExplainer.plot(...)`: renders the standard DPG
- `DPGExplainer.plot_communities(...)`: renders a community-colored DPG

`DPGExplanation` includes `dot`, `graph`, `nodes`, `node_metrics`, `edge_metrics`, `class_boundaries`, and optional `communities`.

#### CLI scripts
The library contains two different scripts to apply DPG:
- `run_dpg_standard.py`: with this script it is possible to test DPG on a standard classification dataset provided by `sklearn` such as `iris`, `digits`, `wine`, `breast cancer`, and `diabetes`.
- `run_dpg_custom.py`: with this script it is possible to apply DPG to your classification dataset, specifying the target class.

#### DPG implementation
The library also contains two other essential scripts:
- `core.py` contains all the functions used to calculate and create the DPG and the metrics.
- `visualizer.py` contains the functions used to manage the visualization of DPG.

#### Output
The DPG output, through `run_dpg_standard.py` or `run_dpg_custom.py`, produces several files:
- the visualization of DPG in a dedicated environment, which can be zoomed and saved;
- a `.txt` file containing the DPG metrics;
- a `.csv` file containing the information about all the nodes of the DPG and their associated metrics;
- a `.txt` file containing the Random Forest statistics (accuracy, confusion matrix, classification report)

## Easy usage
Usage: `python run_dpg_standard.py --dataset <dataset_name> --n_learners <integer_number> --pv <threshold_value> --t <integer_number> --model_name <str_model_name> --dir <save_dir_path> --plot --save_plot_dir <save_plot_dir_path> --attribute <attribute> --communities --clusters --threshold_clusters <float> --class_flag --seed <int>`
Where:
- `dataset` is the name of the standard classification `sklearn` dataset to be analyzed;
- `n_learners` is the number of base learners for the Random Forest;
- `pv` is the threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees;
- `t` is the decimal precision of each feature;
- `model_name` is the name of the `sklearn` model chosen to perform classification (`RandomForestClassifier`,`BaggingClassifier`,`ExtraTreesClassifier`,`AdaBoostClassifier` are currently available);
- `dir` is the path of the directory to save the files;
- `plot` is a store_true variable which can be added to plot the DPG;
- `save_plot_dir` is the path of the directory to save the plot image;
- `attribute` is the specific node metric which can be visualized on the DPG;
- `communities` is a store_true variable which can be added to visualize communities on the DPG;
- `clusters` is a store_true variable which can be added to visualize clusters on the DPG;
- `threshold_clusters` is the threshold used to detect ambiguous nodes in clusters;
- `class_flag` is a store_true variable which can be added to highlight class nodes;
- `seed` controls the random split.
  
Disclaimer: `attribute`, `communities`, and `clusters` are mutually exclusive: DPG supports just one visualization mode at a time.

The usage of `run_dpg_custom.py` is similar, but it requires another parameter:
- `target_column`, which is the name of the column to be used as the target variable;
- while `ds` is the path of the directory where the dataset is.

#### Example `run_dpg_standard.py`
Some examples can be appreciated in the `examples` folder: https://github.com/Meta-Group/DPG/tree/main/examples

In particular, the following DPG is obtained by transforming a Random Forest with 5 base learners, trained on Iris dataset.
The used command is `python run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --dir examples --plot --save_plot_dir examples`.
<p align="center">
  <img src="https://github.com/Meta-Group/DPG/blob/main/dpg_image_examples/iris_bl5_perc0.001_dec2.png" width="800" />
</p>

The following visualizations are obtained using the same parameters as the previous example, but they show two different metrics: _Community_ and _Betweenness centrality_.
The used command for showing communities is `python run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --dir examples --plot --save_plot_dir examples --communities`.
<p align="center">
  <img src="https://github.com/Meta-Group/DPG/blob/main/dpg_image_examples/iris_bl5_perc0.001_dec2_communities.png" width="800" />
</p>

The used command for showing a specific property is `python run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --dir examples --plot --save_plot_dir examples --attribute "Betweenness centrality" --class_flag`.
<p align="center">
  <img src="https://github.com/Meta-Group/DPG/blob/main/dpg_image_examples/iris_bl5_perc0.001_dec2_Betweennesscentrality.png" width="800" />
</p>

***
## Citation
If you use this for research, please cite. Here is an example BibTeX entry:

```bibtex
@inproceedings{arrighi2024dpg,
  title={Decision Predicate Graphs: Enhancing Interpretability in Tree Ensembles},
  author={Arrighi, Leonardo and Pennella, Luca and Marques Tavares, Gabriel and Barbon Junior, Sylvio},
  booktitle={World Conference on Explainable Artificial Intelligence},
  pages={311--332},
  year={2024},
  isbn = {978-3-031-63797-1},
  doi = {10.1007/978-3-031-63797-1_16},
  publisher = {Springer Nature Switzerland},
}
