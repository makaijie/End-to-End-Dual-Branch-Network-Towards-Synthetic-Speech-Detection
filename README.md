# End-to-End Dual-Branch Network Towards Synthetic Speech Detection  

### Prerequisites

- NVIDIA GPU+CUDA CuDNN
- Install Torch1.8 and dependencies

### Training and Test Details
- Please adjust the file location before training and testing;
- Data Preparation
  - Change the `Feature Engineering/CQT/cqt_extract.py`, `Feature Engineering/LFCC/extract_lfcc.m` and `Feature Engineering/LFCC/reload_data.py`
  - Run the `Feature Engineering/CQT/cqt_extract.py`, `Feature Engineering/LFCC/extract_lfcc.m` and `Feature Engineering/LFCC/reload_data.py`

- When you train the network
  - Change the `dual-branch_sum_loss.py` or `dual-branch_alternative_loss.py`
  - Run the `dual-branch_sum_loss.py` or `dual-branch_alternative_loss.py`

- When you test the network 
  - Change the `Result_sum_loss/test_dual.py` or `Result_alternative_loss/test_dual.py`
  - Run the `Result_sum_loss/test_dual.py` or `Result_alternative_loss/test_dual.py`

### Acknowledgements
The code of this work is adapted from https://github.com/yzyouzhang/AIR-ASVspoof, https://github.com/yzyouzhang/Empirical-Channel-CM and https://github.com/joaomonteirof/e2e_antispoofing.
