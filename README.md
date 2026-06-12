# FBSDiff++: Improved Frequency Band Substitution of Diffusion Features for Efficient and Highly Controllable Text-Driven Image-to-Image Translation
[IJCV 2026] Official code of the paper "FBSDiff++: Improved Frequency Band Substitution of Diffusion Features for Efficient and Highly Controllable Text-Driven Image-to-Image Translation". [Paper link](https://arxiv.org/pdf/2601.19115)

# Introduction
With large-scale text-to-image (T2I) diffusion models achieving significant advancements in open-domain image creation, increasing attention has been focused on their natural extension to the realm of text-driven image-to-image (I2I) translation, where a source image acts as visual guidance to the generated image in addition to the textual guidance provided by the text prompt. We propose FBSDiff, a novel framework adapting off-the-shelf T2I diffusion model into the I2I paradigm from a fresh frequency-domain perspective. Through dynamic frequency band substitution of diffusion features, FBSDiff realizes versatile and highly controllable text-driven I2I in a plug-and-play manner (without need for model training, fine-tuning, or online optimization), allowing appearance-guided, layout-guided, and contour-guided I2I translation by progressively substituting low-frequency band, mid-frequency band, and high-frequency band of latent diffusion features, respectively. In addition, FBSDiff flexibly enables continuous control over I2I correlation intensity simply by tuning the bandwidth of the substituted frequency band. To further promote image translation efficiency, flexibility, and functionality, we propose FBSDiff++ which improves upon FBSDiff mainly in three aspects: (1) accelerate inference speed by a large margin (8.9$\times$ speedup in inference) with refined model architecture; (2) improve the Frequency Band Substitution module to allow for input source images of arbitrary resolution and aspect ratio; (3) extend model functionality to enable localized image manipulation and style-specific content creation with only subtle adjustments to the core method. Extensive qualitative and quantitative experiments verify superiority of FBSDiff++ in I2I translation visual quality, efficiency, versatility, and controllability compared to related advanced approaches. 

# Model overview
![](imgs/model_overview.jpg "method_overview")
Our previous work proposes FBSDiff, a plug-and-play method adapting pretrained T2I diffusion model to the realm of text-driven I2I translation from the perspective of dynamic frequency band substitution. FBSDiff comprises an inversion trajectory, a reconstruction trajectory, and a sampling trajectory. An FBSDiff module is designed and inserted in between the reconstruction trajectory and sampling trajectory, transplanting feature frequency band to guarantee I2I consistency. FBSDiff enables versatile I2I applications and continuous intensity control of I2I consistency by tuning the type and bandwidth of the transplanted frequency band, respectively. 

In contrary, FBSDiff++ improves upon FBSDiff in four aspects: (1) FBSDiff++ dispenses with the reconstruction trajectory by extracting guidance features from the inversion trajectory, and thus accelerates inference speed noticeably; (2) FBSDiff++ swaps the 2D-DCT-based FBS module for the AdaFBS module, which achieves the same goal of dynamic frequency band substitution via two successive 1D-DCT filtering steps along the feature height and width axes, allowing for input images of arbitrary aspect ratio rather than only the square images; (3) FBSDiff++ converts the absolute DCT filtering thresholds to the percentile-based relative ones, enabling the model to be adaptive to input images of arbitrary resolution; (4) FBSDiff++ extends model functionality to enable localized image manipulation and style-specific content creation for input images of arbitrary size.

# Environment
We use Anaconda environment with python 3.8 and pytorch 2.0, which can be built with the following commands: <br />
First, create a new conda virtual environment: <br>
<pre><code>
conda create -n FBSDiff++ python=3.8
</code></pre>
Then, install pytorch using conda: <br>
<pre><code>
conda activate FBSDiff++
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
</code></pre>
Last, install the required packages in the requirements.txt:
<pre><code>
pip install -r requirements.txt
</code></pre>

# Download pre-trained models
Our method requires the pre-trained Stable Diffusion model and the CLIP text encoder. <br />
1. Download the Stable Diffusion v1.5 model checkpoint file **v1-5-pruned-emaonly.ckpt** and put it right into the **"models"** folder. It can be downloaded from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main), or from [GoogleDrive](https://drive.google.com/file/d/1qv_imy7tyjyuq0BSo53KBvfFGEpDe-GA/view?usp=sharing). <br />
2. Download the **clip-vit-large-patch14** and put it right into the **"openai"** folder. It can be downloaded from [here](https://huggingface.co/openai/clip-vit-large-patch14) with the demo code, or manually downloaded file by file from [here](https://huggingface.co/openai/clip-vit-large-patch14/tree/main). We also provide a [DoogleDrive link](https://drive.google.com/file/d/1lgM9uL9CY_LS7eHU77pG5LhVpt99KnpQ/view?usp=sharing) to download it for convenience.

# Run the code
Our model is training-free, you can translate a given source image with a certain text prompt by directly running the following inference script:
(1) for I2I application of derivative image generative achieved using low-FBS mode, run the **DCT_low_FBS_infer.py** script:
<pre><code>
python DCT_low_FBS_infer.py
</code></pre>

