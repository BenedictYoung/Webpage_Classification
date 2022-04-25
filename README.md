#  Webpage Classification 

## 1.Introduction
There is a dataset contains webpages collected from computer science departments of various universities.  This project is about learning classifiers to predict the type of webpage from the text.

Since the data in the dataset contains some errors and saved in various encoding format, I did some data cleaning to improve data quality and utility(including delete 2 unrecognized pages). This project is based on cleaned data, which can be found in [./webkb/](https://github.com/BenedictYoung/Webpages_Classification/tree/main/webkb). 

If you want to access the original dataset, you may go to: [link](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/). 

## 2.Dataset
All webpages are labeled into the following 7 target categories:(cleaned/original)

<center>
| Categories|  Cleaned   | Original |
|-----------|------------|----------|
| student   | 1641       | 1641     |
| staff     | 136        | 137      |
| department| 182        | 182      |
| course    | 930        | 930      |
| project   | 504        | 504      |   
| other     | 3763       | 3764     |
</center>


And, the data is divided by universities:(cleaned/original)


<center>
| Categories|  Cleaned   | Original |
|-----------|------------|----------|
| Cornell   | 867        | 867      |
| Texas     | 827        | 827      |
| Washington| 1204       | 1205     |
| Wisconsin | 1263       | 1263     |
| Miscellaneous   | 4119        | 4120      |   
</center>

# 3.Methodology 
Instead of treating the HTML format texts as structured data, I tend to treat webpage as plain text. Therefore, I implemented the pre-trained BERT model to solve this task. After applying some specific task-oriented "fine-tuning", my proposed method could achieve about 94% prediction accuracy. 

# 4.Description

<center>
Table 1: Description of files and documents

| File name      |  Description   |
|---------|--------|
| webkb/  | dataset|
| pretrained/   | configurations of model |
| get_data.py | reading data from 'webkb/'  |
| functions.py   | necessary functions  |
|bert.py | main code|
|poster.pdf| poster|
</center>
