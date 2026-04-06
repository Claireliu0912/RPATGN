# RPATGN

The implementation of "**Role Perceptual Augmented Temporal Graph Network for Related-party Transaction Detection**" (AAAI 2026). 

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/41240)
[Poster](https://underline.io/lecture/138944-role-perceptual-augmented-temporal-graph-network-for-related-party-transaction-detection)
[Supplementary](https://github.com/Claireliu0912/RPATGN/blob/main/script/supplementary.pdf)


## Environment Setup

- Python 3.7.11
- PyTorch 1.13.1
- torch-geometric 2.3.1
- numpy, scikit_learn, tqdm

## Datasets

We conducted experiments on four real-world financial datasets:
- **RPT**: The dataset was collected from regularly disclosed RPT data in China's nationwide financial market from January 2015 to December 2021. It includes profiles of listed companies and related parties, RPT operation information, and financial exchanges involved in these transactions. 
- **Elliptic**: Maps bitcoin transactions to real entities belonging to licit categories versus illicit ones. Nodes represent transactions, edges represent flow of Bitcoins.
- **Bitcoin OTC**: Who-trusts-whom network of people who trade using Bitcoin on the Bitcoin OTC platform. Nodes represent Bitcoin users, edges represent ratings between users.
- **Bitcoin Alpha**: Created in the same way as Bitcoin OTC, but users and ratings come from a different trading platform.

Processed data for Bitcoin OTC and Bitcoin Alpha are provided in the `data/` folder.

## Running the Code

```shell
python main.py --dataset otc --nhid 32 --nout 32 --nb_window 5
python main.py --dataset alpha --nhid 32 --nout 32 --nb_window 5
```

## Project Structure

- `data/`: Dataset files (Bitcoin OTC and Bitcoin Alpha provided)
- `script/`: Model implementations

## Citation
If you find this project helpful, please cite our paper:
```shell
@inproceedings{liu2026role,
  title={Role Perceptual Augmented Temporal Graph Network for Related-party Transaction Detection},
  author={Liu, Xin and Yu, Yuanhang and Zhu, Peng and Cheng, Dawei and Jiang, Changjun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={45},
  pages={38943--38951},
  year={2026}
}
```

## Acknowledgements
The code is built on [HTGN](https://github.com/marlin-codes/HTGN).
