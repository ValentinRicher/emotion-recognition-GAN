# Emotion Recognition with semi-supervised GAN

The goal of this work was to assemble two models used to depict emotions on faces. These models are Action Units and Valence Arousal. <br/>
Action Units are facial muscle movements. The combination of these Action Units can be interpreted as an emotion. 
Valence Arousal is a 2D continuous scale where Valence represents how positive or negative a person is when feeling an emotion and Arousal represents how excited or calm the person is. <br/>
Now the possibility to run the code with the facemotion repository ! 

# To Do

- Add metrics for the real or fake images ✅
- Connect the GAN repo with the dataset repo to create automatic downloading ✅
- Re-organize the facemotion.py to the same level as other .py files ✅
- Use Google Cloud Platform ❌ -> impossible to use a GPU without paying
- Use Google Colab ❌ -> impossible to download the dataset quickly
- Re-organize the code to make it more modular and easy to test different parameters :
  - Create a file for the definition of the architecture
  - Create a config file
- Create an architecture for 64*64 images
- Create an architecture for 96*96 images


# Acknowledgments

Parts of the code from https://github.com/gitlimlab/SSGAN-Tensorflow
