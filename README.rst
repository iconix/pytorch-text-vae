Based on code from Sean Robertson, adapted to word level as in the original paper

To get a saved model, and preprocessed data, download and unzip the file (~1.1GB) at this link
``https://drive.google.com/file/d/0Bzz1g90lrPKNZTVtSHFDM0t0cG8/view?usp=sharing``

Next, run this command to unzip the pretrained models and data

tar xzf stored_pytorch_text_vae_info.tar.gz


Usage:

python interpolate.py -1 "it had taken years to believe" -2 "but it was all lies at the end" -t .01 -s saved_vae.pt


Example output:

('(s0)', u'it had taken years to believe')
('(z0)', ' it had taken time to his')
('  .)', ' it had my hands to his')
('  .)', ' but it was mad at his end')
('  .)', ' but it was nt mad at his end')
('(z1)', ' but it was all her at the end')
('(s1)', u'but it was all lies at the end')
