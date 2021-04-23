from matplotlib import pyplot as plt
import argparse
import pickle as pkl
import os
import sys

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--rep_num', type=int, default=1000)

args = parser.parse_args()
rep_num = args.rep_num

with open('lightgbm_%d.pkl' % rep_num, 'rb') as f:
    results = pkl.load(f)

plt.figure(figsize=(10, 5))
labels = results.keys()
plt.boxplot([results[key] for key in labels], labels=labels)
plt.violinplot([results[key] for key in labels])
plt.xticks(rotation=30)
plt.ylabel('Importance', fontsize=24)
plt.subplots_adjust(top=0.97, right=0.965, left=0.10, bottom=0.25)
plt.show()
