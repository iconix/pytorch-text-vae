An partial reimplementation of "Generating Sentences From a Continuous Space", Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio (``https://arxiv.org/abs/1511.06349``). 

Based on code from Sean Robertson (``@spro``) ``https://github.com/spro/pytorch-text-vae``, adapted to word level as in the original paper.

This code doesn't implement iterative conditional modes for sampling, or several other details of the original paper. The resulting interpolations in this recreation seem less grammatically accurate than those presented in the paper, though the overall result is similar.

To get a saved model, and preprocessed data, download the file (~1.1GB) at this link

``https://drive.google.com/file/d/0Bzz1g90lrPKNZTVtSHFDM0t0cG8/view?usp=sharing``

Next, run this command to unzip the pretrained models and preprocessed data

``tar xzf stored_pytorch_text_vae_info.tar.gz``

The pretrained model was trained on the Book Corpus dataset (``http://yknzhu.wixsite.com/mbweb``).


Sampling Usage:

``python interpolate.py -1 "it had taken years to believe" -2 "but it was all lies at the end" -t .01 -s saved_vae.pt``


Example output:

    ('(s0)', u'it had taken years to believe')

    ('(z0)', ' it had taken time to his')
    
    ('  .)', ' it had my hands to his')
    
    ('  .)', ' but it was mad at his end')
    
    ('  .)', ' but it was nt mad at his end')
    
    ('(z1)', ' but it was all her at the end')
    
    ('(s1)', u'but it was all lies at the end')


New Model Training:

``python train.py myfile.txt``

Where ``myfile.txt`` is a text file with one sentence per line. The model will be saved in ``vae.pt``.
