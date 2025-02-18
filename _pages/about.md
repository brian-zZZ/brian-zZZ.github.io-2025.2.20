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

I am currently work at [Huawei Technolgy](https://www.huawei.com/) as an AI engineer in Shanghai, China. My work focuses on the intelligent automated evaluation of Large Language Models (LLMs). I will become a PhD candidate at [School of Software and Microelectronics](https://www.ss.pku.edu.cn/), Peking University (北京大学软件与微电子学院) in the fall semester of 2025. My research interest includes LLMs, multimodality learning, representation learning, and transferability.

I graduated from the [University of Chinese Academy of Sciences](https://www.ucas.ac.cn/) (中国科学院大学) with a Master's degree in Computer Technology, advised by [Yunpeng Cai (蔡云鹏)](https://people.ucas.edu.cn/~caiyunpeng). I earned my bachelor’s degree at [College of Electronics and Information Engineering](https://ceie.szu.edu.cn/), Shenzhen University.

As the first or co-first author, I have published four articles in top-tier journals or JCR Q1-ranked journals with total <a href='https://scholar.google.com/citations?user=P7wwiSMAAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>. I have also presented a conference paper and published an invention patent.


# 🔥 News
- *2025.02*: &nbsp;🎉🎉 My paper [A Transferability-guided Protein-ligand Interaction Prediction Method](https://doi.org/10.1016/j.ymeth.2025.01.019) is accepted by *Methods*.
- *2025.01*: &nbsp;🎉🎉 I have been admitted as a PhD student to the School of Software and Microelectronics, Peking University <img src='./images/pku_ss_logo.png' style='width: 6em;'>.

# 📝 Publications 

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Advanced Science</div><img src='images/Adv.Sci._2023.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[A Multimodal Protein Representation Framework for Quantifying Transferability Across Biochemical Downstream Tasks](https://doi.org/10.1002/advs.202301223)

Fan Hu¹, Yishen Hu¹, **Weihong Zhang**¹, Huazhen Huang, Yi Pan, and Peng Yin
> ¹: These authors contributed equally to this work (the same applies hereafter).

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=P7wwiSMAAAAJ&citation_for_view=P7wwiSMAAAAJ:u5HHmVD_uO8C) <strong><span class='show_paper_citations' data='P7wwiSMAAAAJ:u5HHmVD_uO8C'></span></strong>
[**GitHub Repository**](https://github.com/SIAT-code/MASSA)
- A SOTA multimodal deep learning framework for incorporating ≈1 million protein sequence, structure, and functional annotation (MASSA). A novel optimal-transport-based metric with rich geometry awareness is introduced to quantify the dynamic transferability from the pretrained representation to the related downstream tasks, which provides a panoramic view of the step-by-step learning process.
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">JBHI</div><img src='images/JBHI_2024.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[A Transferability-Based Method for Evaluating the Protein Representation Learning](https://doi.org/10.1109/JBHI.2024.3370680)

Fan Hu¹, **Weihong Zhang**¹, Huazhen Huang, Wang Li, Yang Li, Peng Yin

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=P7wwiSMAAAAJ&citation_for_view=P7wwiSMAAAAJ:9yKSN-GCB0IC) <strong><span class='show_paper_citations' data='P7wwiSMAAAAJ:9yKSN-GCB0IC'></span></strong>
[**GitHub Repository**](https://github.com/SIAT-code/OTMTD)
- A novel quantitative approach for estimating the performance of transferring multi-task pre-trained protein representations to downstream tasks. This transferability-based method is designed to quantify the similarities in latent space distributions between pre-trained features and those fine-tuned for downstream tasks. It encompasses a broad spectrum, covering multiple domains and a variety of heterogeneous tasks. Our experimental results demonstrate a robust correlation between the transferability scores obtained using our method and the actual transfer performance observed.
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">JBHI</div><img src='images/Methods_2023.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Does Protein Pretrained Language Model Facilitate the Prediction of Protein–ligand Interaction?](https://doi.org/10.1016/j.ymeth.2023.08.016)

**Weihong Zhang**, Fan Hu, Wang Li, Peng Yin

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=P7wwiSMAAAAJ&citation_for_view=P7wwiSMAAAAJ:u-x6o8ySG0sC) <strong><span class='show_paper_citations' data='P7wwiSMAAAAJ:u-x6o8ySG0sC'></span></strong>
[**GitHub Repository**](https://github.com/brian-zZZ/PLM-PLI)
- An approach that quantitatively evaluates the impact of protein pretrained language model (PLM) in protein–ligand interaction (PLI) predictions, which allows us to select the optimal PLM for a given downstream task without exhaustively testing each PLM, thus avoiding the costly computational expense. The mechanisms underlying the influence of protein PLMs on PLI tasks are explored. Our observations suggest that pre-training serves as a process of embedding prior knowledge, as evidenced by the increased distinctiveness of feature distributions among different tasks after pre-training.
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Methods</div><img src='images/Methods_2025.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[A Transferability-guided Protein-ligand Interaction Prediction Method](https://doi.org/10.1016/j.ymeth.2025.01.019)

**Weihong Zhang**, Fan Hu, Peng Yin, Yunpeng Cai

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=P7wwiSMAAAAJ&citation_for_view=P7wwiSMAAAAJ:2osOgNQ5qMEC) <strong><span class='show_paper_citations' data='P7wwiSMAAAAJ:2osOgNQ5qMEC'></span></strong>
[**GitHub Repository**](https://github.com/brian-zZZ/Guided-PLI)
- A novel transferability-guided PLI prediction method that maximizes knowledge transfer by deeply integrating protein and ligand representations through a cross-attention mechanism and incorporating transferability metrics to guide fine-tuning. The transferability-guided strategy quantifies transferability from pretraining tasks and incorporates it into the training objective, ensuring the effective utilization of beneficial knowledge while mitigating negative transfer. This strategy provides a paradigm for more comprehensive utilization of pretraining knowledge.
</div>
</div>


# 🎖 Honors and Awards
- *2021.10* Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2021.09* Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 

# 📖 Educations
- *2025.09 - 2022.04 (now)*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2015.09 - 2019.06*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 

# 💬 Invited Talks
- *2021.06*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2021.03*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.  \| [\[video\]](https://github.com/)

# 💻 Internships
- *2019.05 - 2020.02*, [Lorem](https://github.com/), China.
