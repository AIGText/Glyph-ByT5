# Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering


<a href='https://arxiv.org/abs/2403.09622'><img src='https://img.shields.io/badge/Arxiv-2403.09622-red'>
<a href='https://arxiv.org/abs/2406.10208'><img src='https://img.shields.io/badge/Arxiv-2406.10208-red'>
<a href='https://glyph-byt5.github.io/'><img src='https://img.shields.io/badge/Project Page-GlyphByT5-green'>
<a href='https://glyph-byt5-v2.github.io/'><img src='https://img.shields.io/badge/Project Page-GlyphByT5v2-green'></a>


This is the official implementation of Glyph-ByT5 and Glyph-ByT5-v2, introduced in [Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering](https://arxiv.org/abs/2403.09622) and [Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering
](https://arxiv.org/abs/2406.10208). 


## :high_brightness: Highlights

* We identify two crucial requirements of text encoders for achieving accurate visual text rendering: character awareness and alignment with glyphs. To this end, we propose a customized text encoder, Glyph-ByT5, by fine-tuning the character-aware ByT5 encoder using a meticulously curated paired glyph-text dataset.

* We present an effective method for integrating Glyph-ByT5 with SDXL, resulting in the creation of the Glyph-SDXL model for design image generation. This significantly enhances text rendering accuracy, improving it from less than 20% to nearly 90% on our design image benchmark. Noteworthy is Glyph-SDXL's newfound ability for text paragraph rendering, achieving high spelling accuracy for tens to hundreds of characters with automated multi-line layouts.

* We deliver a powerful customized multilingual text encoder, Glyph-ByT5-v2, and a strong aesthetic graphic generation model, Glyph-SDXL-v2, that can support accurate spelling in $\sim10$ different languages

<table>
  <tr>
    <td><img src="inference/assets/teaser/paragraph_1.png" alt="paragraph example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/paragraph_2.png" alt="paragraph example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/paragraph_3.png" alt="paragraph example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/paragraph_4.png" alt="paragraph example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/design_1.png" alt="design example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/design_2.png" alt="design example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/design_3.png" alt="design example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/design_4.png" alt="design example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/scene_1.png" alt="scene example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/scene_2.png" alt="scene example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/scene_3.png" alt="scene example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/scene_4.png" alt="scene example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/cn_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/cn_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/cn_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/cn_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/fr_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/fr_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/fr_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/fr_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/de_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/de_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/de_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/de_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/jp_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/jp_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/jp_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/jp_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/kr_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/kr_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/kr_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/kr_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/es_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/es_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/es_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/es_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/it_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/it_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/it_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/it_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/pt_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/pt_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/pt_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/pt_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/ru_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/ru_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/ru_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/ru_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
</table>


## :wrench: Usage

For a detailed guide on Glyph-SDXL and Glyph-SDXL-v2 inference, see [this folder](inference/).


## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:

```
@misc{liu2024glyphbyt5,
    title={Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering},
    author={Zeyu Liu and Weicong Liang and Zhanhao Liang and Chong Luo and Ji Li and Gao Huang and Yuhui Yuan},
    year={2024},
    eprint={2403.09622},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

and 

```
@misc{liu2024glyphbyt5v2,
    title={Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering}, 
    author={Zeyu Liu and Weicong Liang and Yiming Zhao and Bohan Chen and Ji Li and Yuhui Yuan},
    year={2024},
    eprint={2406.10208},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```