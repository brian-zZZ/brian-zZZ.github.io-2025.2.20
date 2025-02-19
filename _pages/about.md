---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I am currently work at [Huawei Technolgies](https://www.huawei.com/) as an AI engineer in Shanghai, China. My work focuses on the intelligent automated evaluation of Large Language Models (LLMs). I will become a PhD candidate at [School of Software and Microelectronics](https://www.ss.pku.edu.cn/), Peking University (北京大学软件与微电子学院) in the fall semester of 2025. My research interest includes LLMs, multimodality learning, representation learning, and transferability.

I graduated from the [University of Chinese Academy of Sciences](https://www.ucas.ac.cn/) (中国科学院大学) with a Master's degree in Computer Technology, advised by [Yunpeng Cai](https://people.ucas.edu.cn/~caiyunpeng). I earned my bachelor’s degree at [College of Electronics and Information Engineering](https://ceie.szu.edu.cn/), Shenzhen University.

As the first or co-first author, I have published four articles in top-tier journals or JCR Q1-ranked journals with total <a href='https://scholar.google.com/citations?user=P7wwiSMAAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>. I have also presented a conference paper and published an invention patent.


# 🔥 News
- *2025.02*: &nbsp;🎉🎉 My paper [A Transferability-guided Protein-ligand Interaction Prediction Method](https://doi.org/10.1016/j.ymeth.2025.01.019) is accepted by *Methods*.
- *2025.01*: &nbsp;🎉🎉 I have been admitted as a PhD student to the School of Software and Microelectronics, Peking University <img src='./images/pku_ss_logo.png' style='width: 6em;'>.


# 📖 Educations
- *2025.09 -*: PhD Candidate, School of Software and Microelectronics, Peking University <img src='./images/pku_words.png' style='width: 6em;'>.
- *2021.09 - 2024.06*: Master, Shenzhen Institutes of Advanced Technology (SIAT), University of Chinese Academy of Sciences <img src='./images/ucas_words.png' style='width: 6em;'>.
- *2017.07 - 2021.06*: Bachelor, College of Electronics and Information Engineering, Shenzhen University <img src='./images/szu_words.png' style='width: 6em;'>.


# 📝 Publications 

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Advanced Science</div><img src='images/Adv.Sci._2023.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[A Multimodal Protein Representation Framework for Quantifying Transferability Across Biochemical Downstream Tasks](https://doi.org/10.1002/advs.202301223)

Fan Hu¹, Yishen Hu¹, **Weihong Zhang**¹, Huazhen Huang, Yi Pan, and Peng Yin
> **¹: These authors contributed equally to this work** (the same hereafter).

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=P7wwiSMAAAAJ&citation_for_view=P7wwiSMAAAAJ:u5HHmVD_uO8C) <strong><span class='show_paper_citations' data='P7wwiSMAAAAJ:u5HHmVD_uO8C'></span></strong> \| [![**GitHub Repository**](https://img.shields.io/github/stars/SIAT-code/MASSA?style=social&label=Code+Stars)](https://github.com/SIAT-code/MASSA)
- A SOTA multimodal deep learning framework for incorporating ≈1 million protein sequence, structure, and functional annotation (MASSA).
- Academic Impact: This work is promoted by [DrugAI](https://mp.weixin.qq.com/s/YFk9OVoHqAZNwkXpx41LQw).
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">IEEE JBHI</div><img src='images/JBHI_2024.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[A Transferability-Based Method for Evaluating the Protein Representation Learning](https://doi.org/10.1109/JBHI.2024.3370680)

Fan Hu¹, **Weihong Zhang**¹, Huazhen Huang, Wang Li, Yang Li, Peng Yin

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=P7wwiSMAAAAJ&citation_for_view=P7wwiSMAAAAJ:9yKSN-GCB0IC) <strong><span class='show_paper_citations' data='P7wwiSMAAAAJ:9yKSN-GCB0IC'></span></strong> \| [![**GitHub Repository**](https://img.shields.io/github/stars/SIAT-code/OTMTD?style=social&label=Code+Stars)](https://github.com/SIAT-code/OTMTD)
- A novel quantitative approach for estimating the performance of transferring multi-task pre-trained protein representations to downstream tasks.
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Methods</div><img src='images/Methods_2023.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Does Protein Pretrained Language Model Facilitate the Prediction of Protein–ligand Interaction?](https://doi.org/10.1016/j.ymeth.2023.08.016)

**Weihong Zhang**, Fan Hu, Wang Li, Peng Yin

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=P7wwiSMAAAAJ&citation_for_view=P7wwiSMAAAAJ:u-x6o8ySG0sC) <strong><span class='show_paper_citations' data='P7wwiSMAAAAJ:u-x6o8ySG0sC'></span></strong> \| [![**GitHub Repository**](https://img.shields.io/github/stars/brian-zZZ/PLM-PLI?style=social&label=Code+Stars)](https://github.com/brian-zZZ/PLM-PLI)
- An approach that quantitatively evaluates the impact of protein pretrained language model (PLM) in protein–ligand interaction (PLI) predictions, which allows us to select the optimal PLM for a given downstream task without exhaustively testing each PLM, thus avoiding the costly computational expense. The mechanisms underlying the influence of protein PLMs on PLI tasks are explored.
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Methods</div><img src='images/Methods_2025.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[A Transferability-guided Protein-ligand Interaction Prediction Method](https://doi.org/10.1016/j.ymeth.2025.01.019)

**Weihong Zhang**, Fan Hu, Peng Yin, Yunpeng Cai

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=P7wwiSMAAAAJ&citation_for_view=P7wwiSMAAAAJ:2osOgNQ5qMEC) <strong><span class='show_paper_citations' data='P7wwiSMAAAAJ:2osOgNQ5qMEC'></span></strong> \| [![**GitHub Repository**](https://img.shields.io/github/stars/brian-zZZ/Guided-PLI?style=social&label=Code+Stars)](https://github.com/brian-zZZ/Guided-PLI)
- A novel transferability-guided PLI prediction method that maximizes knowledge transfer by deeply integrating protein and ligand representations through a cross-attention mechanism and incorporating transferability metrics to guide fine-tuning.
</div>
</div>

# 📝 Publication List

- <span class='paper-badge'>ISBRA 2024</span> **A Transferability-guided Protein-ligand Interaction Prediction Method**<br>
  _<u><b>Weihong Zhang</b></u>, Fan Hu, Peng Yin, Yunpeng Cai_<br>
  _The 20th International Symposium on Bioinformatics Research and Application_, 2024 (CCF-C)<br>
  <span class='paper-asset'><a href="https://doi.org/10.1016/j.ymeth.2025.01.019">Page</a></span> <span class='paper-asset'><a href="https://bio.csu.edu.cn/ISBRA2024/ISBRA2024_files/ISBRA2024_Schedule.pdf">PDF</a></span> <span class='paper-asset'><a href="https://github.com/brian-zZZ/Guided-PLI">Code</a></span>

- <span class='paper-badge'>Invention Pattern</span> **An Evaluation Method and System for Protein Representation Learning Based on Quantitative Transferability (一种基于可迁移性定量的蛋白质表示学习评估方法及系统)**<br>
  _Fan Hu, <u><b>Weihong Zhang</b></u>, Peng Yin_<br>
  _China Invention Pattern, CN117637034A_, 2024<br>
  <span class='paper-asset'><a href="http://epub.cnipa.gov.cn/patent/CN117637034A">Page</a></span>


# 🗣 Conferences
- *2024.07*: The 20th International Symposium on Bioinformatics Research and Application (ISBRA 2024), Oral, Kunming, China.


# 💻 Internships
- *2024.07 - present*: <a href="https://www.huawei.com/"><img src='./images/huawei.png' width="23pt"></a> AI Engineer, Huawei Technologies Co., Ltd., Shanghai, China.
- *2021.06 - 2021.08*: <a href="https://www.siat.ac.cn/"><img src='./images/siat.jpeg' width="23pt"></a> Research Intern, SIAT, Chinese Academy of Sciences, Shenzhen, China.


# 🎖 Honors and Awards
- *2024.07*: Merit Student of University of Chinese Academy of Sciences.
- *2024.04*: Outstanding Communist Youth League Member of Chinese Academy of Sciences (Guangzhou).
- *2024.01*: SIAT President's Scholarship - Excellence Award.
- *2023.02*: Outstanding Student at the BIT Center, SIAT, Chinese Academy of Sciences.
- *2021.06*: Outstanding Graduate of the School of Electronic and Information Engineering, Shenzhen University.
- *2020.12*: Shenzhen University Academic Star Scholarship.
- *2020.10*: National Endeavor Scholarship.
- *2020.07*: Outstanding Communist Youth League Member of Shenzhen University.
- *2019.12*: Shenzhen University Public Service Star Scholarship.
- *2019.06*: Outstanding Volunteer Officer of Shenzhen University.
- *2018.12*: Shenzhen University Public Service Star Scholarship.
- *2018.10*: National Endeavor Scholarship.

<script type='text/javascript' id='clustrmaps' src='//cdn.clustrmaps.com/map_v2.js?cl=ffffff&w=300&t=tt&d=KKVQwhlxW3eaKbtrI8kx7IgDAyBI9TlARuKNBXIu9g0'></script>
<!-- <script type="text/javascript" id="clustrmaps" src="//clustrmaps.com/map_v2.js?d=KKVQwhlxW3eaKbtrI8kx7IgDAyBI9TlARuKNBXIu9g0&cl=ffffff&w=600"></script> -->
