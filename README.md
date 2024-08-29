<div align="center">
<h2 align="center">
   <b>RuleAlign Dataset</b>
</h2>
<div>
<a target="_blank" href="https://scholar.google.com.sg/citations?user=GqZfs_IAAAAJ&hl=en">Fangyuan&#160;Yu</a><sup>1 2</sup>
</div>
<sup>1</sup>Temus&#160&#160&#160</span>
<!-- <sup>2</sup>Stanford University</span> -->
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<br/>
<div align="center">
    <a href="https://arxiv.org/abs/xxxx" target="_blank">
</div>
</div>
<h3 align="center">
<b>:fire: Dataset and code will be released soon</b>
</h3>

## :books: RuleEval Dataset

RuleAlign is a dataset designed to evaluate rule-based alignment of language models. It accompanies the paper "Iterative Graph Reasoning" and contains:

- 5 distinct rule-based scenarios
- 200 training queries per scenario (low-data learning scenario)
- 100 test queries per scenario
- Weak annotations provided for each query

## :new: Updates
- [08/2024] [Arxiv paper](https://arxiv.org/abs/2408.03615) released.
- [08/2024] RuleAlign dataset announced.

## :gear: Evaluation

To run the evaluation on the RuleAlign dataset, use the following command:

```shell
python -m script.eval --model_name xxx
```

Replace `xxx` with the name of the model you want to evaluate.

## :hugs: Citation
If you find this dataset useful for your research, please kindly cite our paper:

```
@misc{yu2024iterativegraphreasoning,
      title={Iterative Graph Reasoning}, 
      author={Fangyuan Yu},
      year={2024},
      eprint={2408.03615},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## :mailbox: Contact

For any questions or issues regarding the RuleEval dataset, please contact the corresponding author.
