This repo contains some of the codes for the following paper [Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards. Code and Data.](https://arxiv.org/abs/2008.02693)

# Special Note:
1. This dataset is much bigger than the one used on ECCV 2020. The larger one has almost 1M images while the other one contains only about half of it (even though you might find 993K in the paper).
2. The evaluation codes are now adopted from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
3. Because of the two reasons above, we now should have better CIDEr scores. However, the other scores might be lower. We will try to update the scores soon.

# Codes:
Now this repo only contains codes for [SAT](https://arxiv.org/abs/1502.03044), [BUTD](https://arxiv.org/abs/1707.07998) and [CNN-C](https://arxiv.org/abs/1711.09151) as was written in the paper.

evalcap folder can be downloaded from [here](https://drive.google.com/file/d/1Y2h7Q_3l3DOR7WKXk_N5SKdAkcHUxQ6F/view?usp=sharing).

To run the code for training, do sh train.sh. To test, sh test.sh

I kept having bad results for CNN-C model, with all the generations in the val set be the same. I had the same issue when I tried to adopt from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). This never happened before when I ran the experiments for the ECCV paper. I really appreciate if anyone find the reason why this happened.
# Dataset:
To get the preprocessed data, use [this](https://drive.google.com/drive/folders/1cgdHt8AlBukmPhuSzUTPszYPXAYmg6gy?usp=sharing) or email: Xuewen Yang @ xuewen.yang@stonybrook.edu if you need the raw data.

For other issues, please create an issue on this repo.

If you want to download the original dataset (some data might be missing), you can:
1. First download the json file from [here](https://drive.google.com/file/d/1IqsiHFF3SkU6NHuaLcGByMN0-HF01dFD/view?usp=sharing).
2. Then use wget or other download scripts. For example, wget https://n.nordstrommedia.com/id/sr3/58d1a13f-b6b6-4e68-b2ff-3a3af47c422e.jpeg
Remember to ignore anything after .jpeg in the url to get high resolution images, otherwise, very low resolution images are downloaded.
3. Sometimes the description is no longer available, we can retrieve it from the 'detail_info' part.

# License:
1. The dataset is under license in the LICENSE file.
2. No commercial use.

# Citation:
If you use this data, please cite:
```
@inproceedings{XuewenECCV20Fashion,
Author = {Xuewen Yang and Heming Zhang and Di Jin and Yingru Liu and Chi-Hao Wu and Jianchao Tan and Dongliang Xie and Jue Wang and Xin Wang},
Title = {Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards},
booktitle = {ECCV},
Year = {2020}
}
```

