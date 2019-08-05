# Real-Time High-Quality Specular Highlight Removal Using Efficient Pixel Clustering

by Antonio Souza, Márcio Macedo, Verônica Nascimento and Bruno Oliveira.

### Introduction

This is a C++/OpenCV/CUDA application for real-time specular highlight removal in still images. Technical details are provided in our [SIBGRAPI 2018 paper.](https://doi.org/10.1109/SIBGRAPI.2018.00014) 

To enable real-time specular highlight removal using GPU, uncomment the line #define REMOVE_SPECULAR_HIGHLIGHT_USING_CUDA in include/SpecularHighlightRemoval/useCUDA.h.

## Citation

The provided source codes are in public domain and can be downloaded for free. If this work is useful for your research, please consider citing:

  ```shell
  @inproceedings{Macedo2018,
  author={Souza, Antonio and Macedo, Marcio and Nascimento, Veronica and Oliveira, Bruno},
  title={Real-Time High-Quality Specular Highlight Removal Using Efficient Pixel Clustering},
  booktitle={Proceedings of the 31st Conference on Graphics, Patterns and Images (SIBGRAPI)},
  year={2018},
  pages={56-63},
  doi={10.1109/SIBGRAPI.2018.00014},
  publisher={IEEE}
  }
  ```
