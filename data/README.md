# How to check all data is correct in this folder?

Run this command and verify that the output matches:
```
> shasum *.json *.csv *.dic
4754550d696da35558fc1edc18ff38f6992c5ece  en_lexeme_prob.json
b514b15f69ada834812e54ee47c8b9b229ff4cd6  13428_2013_403_MOESM1_ESM.csv
5205586aab9cc86ac3c435609d7b7e5f05287db6  AoA_51715_words.csv
bca1841ee9cf82e78559b57af8252ec8715ba0bb  WordprevalencesSupplementaryfilefirstsubmission.csv
3a047b4d4b113565e615b95a3bde47d3d2ad590d  mrc2.csv
6edcb576a6373a84c4c507819425a7760c116b25  hyph_en_US.dic
```

If for whatever reason this doesn't match, check download_logs.txt for issues with downloading files.
If that file is not present at all - you probably forgot to run pip install - please install the library first.