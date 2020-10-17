# Singing-voice-speration-with-U-Net
The purpose of this project is to decompose a music audio signal into its vocal and backing track components. Once decomposed, clean vocal or backing signals are useful for other MIR tasks such as singer identification, lyric transcription, and karaoke application [1]. While there are traditional methods for doing source separation of musical audio, deep learning models have emerged as powerful alternatives. In [1], the authors propose a method for decomposing music audio signals into a backing track using modified convolutional neural networks called U-Net [2]. While U-Net was developed for biomedical imaging [2,3], this architecture can be used in the context of singing voice separation. The results of experimentation in [1] shows that U-Net achieves better performance than the other state-of-the-art deep learning technique. We find this paper [1] would be interesting to prove, especially from a video in [4], where the presenter uses U-Net to show how clean this technique does voice separation.

---

### UNet Architecture
+ is a U-shape Convolutional Neural Networks designed for Biomedical Image Segmentation
+ (in the context of MIR) allows recreating low-level detail for high quality audio reproduction
+ is a type of fully convolutional network (FCN), which has only convolutional layers (i.e. layers are not fully connected).
+ has two parts:
    1. <b>Encoder</b> that learns highly abstract representation of the input image.
    2. <b>Decoder</b> takes encoder's input and maps it into segmented groundtruth data (e.g. by transposing convolutions and unpooling).
  
### Encoder: 
+ is a set of convolutional layers that reduce the inputs dimensionality while preserving prevalent information
+ applies Localization:
    + combines high resolution features from the contracting path with the upsampled output
    + each successive convolution layers learn to assemble a more precise output
+ has 6 blocks of:
+ 5x5 convolutions (filters) with stride = 2
    + stride controls how the filter convolves around the input volume
+ the first layer has 16 filters, which doubled at each downsampling layer (and half number of channels)
+ batch normalization
+ leaky ReLU activations with leakiness (alpha = 0.2)
+ each downsampling (maxpooling) step doubles the number of features map
        
### Decoder:
+ recreates the input from the contracted representation by encoder
+ is symmetric to the Encoder (i.e. has the same number of filters, sizes, strides, and output dimensions)
+ has 6 blocks of:
    + 50% dropout (p = 0.5) for the first three layers
    + 5x5 deconvolutions with stride = 2 (i.e. halved layer, double number of channels)
    + plain ReLU
    + sigmoid activation in the final layer
    
+ 128 size for Mini-batch training, with ADAM optimizer

<br>

![Image](img/u-net-architecture.png)

---
### Cool Reads
+ U-Net tutorial: https://www.youtube.com/watch?v=azM57JuQpQI&t=552s&ab_channel=DigitalSreeni
+ Basic project structure: https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600
    
---
<br>
### References:
<sup>1</sup> ![Singing Voice Separation With Deep U-NET Convolutiinal Networks](https://openaccess.city.ac.uk/id/eprint/19289/1/7bb8d1600fba70dd79408775cd0c37a4ff62.pdf)<br>
<sup>2</sup> ![U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)<br>
