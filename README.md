# Emotion Recognition with semi-supervised GAN

The goal of this work was to assemble two models used to depict emotions on faces. These models are Action Units and Valence Arousal. <br/>
Action Units are facial muscle movements. The combination of these Action Units can be interpreted as an emotion. 
Valence Arousal is a 2D continuous scale where Valence represents how positive or negative a person is when feeling an emotion and Arousal represents how excited or calm the person is. <br/>
Now the possibility to run the code with the facemotion repository ! 

# Prerequisites 

- Python 2.7
- Virtual environment manager :
  - [virtualenv](https://virtualenv.pypa.io/en/latest/)
  - [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
  
- [Git LFS](https://git-lfs.github.com) <br/>
Git LFS is necessary to download the images contained in the FaceMotion dataset

# Setup

1. Clone the project to your environment :
    ```
    git clone https://github.com/ValentinRicher/emotion-recognition-GAN.git
    ```

2. Create the virtual environment : 
  - with virtualenv
    ```
    virtualenv <venv-name>
    ```
  - or with virtualenvwrapper
    ```
    mkvirtualenv <venv-name>
    ```

3. Activate your virtual environment :
  - with virtualenv
    ```
    source <venv-name>/bin/activate
    ```
  - with virtualenvwrapper
    ```
    workon <venv-name>
    ```

4. Install the libraries :
  - if you use a GPU (recommended)
    ```
    pip install -R gpu-requirements.txt
    ```
  - if you use a CPU
    ```
    pip install -R requirements.txt
    ```

# Usage

- Download the FaceMotion dataset
    ```
    python download.py --model xx --img_size yy
    ```
    This will download the images from the FaceMotion dataset into a ./datasets/facemotion directory if not already done and create the h5py files with the good labels and image sizes.


- Train the model
    ```
    python trainer.py --model xx --img_size yy
    ```
    
- Evaluate the model
    - if you want to test a specific model (here model-201)
      ```
      python evaler.py --checkpoint_path ckpt_p
      ```
      `ckpt_p` should be like : `BOTH-is_32-bs_64-lr_1.00E-04-ur_5-20190217_145915/train_dir/model-201`
    
    - if you want to test the last model saved 
      ```
      python evaler.py --train_dir tr_d
      ```
      `tr_d` should be like : `BOTH-is_32-bs_64-lr_1.00E-04-ur_5-20190217_145915/train_dir/`
    - if you want to test all the models in train_dir
      ```
      python evaler.py --train_dir tr_d --all
      ```

> For the moment it is only possible to work with 32\*32 pixels images because the model architecture for 64\*64 and 96\*96 are not ready yet

# Results

#### [default model] -> model : BOTH / image size : 32 / batch_size : 64 / learning rate : 1e-4 / update rate : 5 / 1 000 000 epochs

> In the following grid of images, 1 image is generated for each epoch

Images created by the Generator during training :
![train fake images](/images/grid_train_fake_0.png)

Images created by the Generator during testing :
![test fake images](/images/grid_test_fake_0.png)

Real images used for training the Discriminator :
![train real images](/images/grid_train_real.png)

Real images used for testing the Discriminator :
![test real images](/images/grid_test_real.png)

# To Do

- Add metrics for the real or fake images ✅
- Connect the GAN repo with the dataset repo to create automatic downloading ✅
- Re-organize the facemotion.py to the same level as other .py files ✅
- Use Google Cloud Platform ❌ -> impossible to use a GPU without paying
- Use Google Colab ❌ -> impossible to download the dataset quickly
- Config file to YAML ✅
- Add a file to get the info stored in events files created for TensorBoard ✅
- Do a notice to explain the project and how to use it 
- Create an architecture for 64*64 images
- Create an architecture for 96*96 images
- Add an early stopping possibility


# Acknowledgments

Parts of the code from https://github.com/gitlimlab/SSGAN-Tensorflow
