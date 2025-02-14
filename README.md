# UMDAOSC650
University of Maryland's AOSC 650 course: Neural Networks for the Physical Sciences

In this course, we will learn the fundamentals of neural networks,
from the basics of a neuron to more advanced architectures, and how these tools can be applied to
the physical sciences. Important considerations for data preprocessing specific to the physical
sciences will be discussed, along with how to evaluate the skill, uncertainty, and confidence of neural
networks with specific relevance to the physical sciences.

The course syllabus is available [here](https://docs.google.com/viewer?url=https://docs.google.com/document/d/1-sdX_Ngq0N21pWYkajzWeO2eLgn-sFBkEOLWvZ6-wQI/export?format=pdf).

## Installing repository locally

To use this repository for the course, fork your own copy of the repository.

Then, your fork of the repository can be downloaded to your local machine using ``git`` on your terminal:

``git clone https://github.com/$YOUR_GITHUB_USERNAME$/UMDAOSC650.git``

To install the necessary python environment, using the provided yaml file is strongly recommended. 

Run the following command from within the ``UMDAOSC650`` directory to install:

``conda env create -f keras-tf.yml``

Once installed, the python environment can be activated by running:

``conda activate keras-tf-v2025``

To deactivate the python environment, run:

``conda deactivate``
