# Fashion_Captioning
ECCV2020 paper: Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards. Code and Data.
To get the data, email: xuewen.yang@stonybrook.edu

The dataset is under license in the LICENSE file.

Download the json file from https://drive.google.com/file/d/1IqsiHFF3SkU6NHuaLcGByMN0-HF01dFD/view?usp=sharing

To get the data, use wget: for example, wget https://n.nordstrommedia.com/id/sr3/58d1a13f-b6b6-4e68-b2ff-3a3af47c422e.jpeg 
Ignore the url after .jpeg to get high resolution image.

{'id': 122349, 'images': [{'color': 'Gold', '0': 'https://n.nordstrommedia.com/id/sr3/58d1a13f-b6b6-4e68-b2ff-3a3af47c422e.jpeg?crop=pad&pad_color=FFF&format=jpeg&w=60&h=90'}], 'title': 'chain link jeans cuff bracelet', 'description': 'subtly futuristic and edgy this liquid metal cuff bracelet is shaped from sculptural rectangular link', 'detail_info': 'DETAILS & CARE\nSubtly futuristic and edgy, this liquid-metal cuff bracelet is shaped from sculptural rectangular links.\n3/8" width\nGoldtone plate\nImported\nItem #6023768\nHelpful info:\nKeep jewelry away from water and chemicals; remove during physical activities; store separately in a soft pouch.', 'categoryid': 30, 'category': 'bracelet', 'attr': ['nah', 'nah', 'nah', 'nah', 'nah', 'nah', 'metal', 'cuff', 'nah', 'nah', 'shaped', 'nah', 'nah', 'rectangular', 'link'], 'attrid': [0, 0, 0, 0, 0, 0, 770, 282, 0, 0, 654, 0, 0, 196, 1]}

Sometimes the description is no longer available, we can retrieve it from the 'detail_info' part.

If you use this data, please cite:

@inproceedings{XuewenECCV20Fashion,
Author = {Xuewen Yang and Heming Zhang and Di Jin and Yingru Liu and Chi-Hao Wu and Jianchao Tan and Dongliang Xie and Jue Wang and Xin Wang},
Title = {Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards},
booktitle = {ECCV},
Year = {2020}
}

