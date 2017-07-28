# Readme

## Environment

The project was written in a Python 2 anaconda environment. 

Full list of installed packages and their version:

appnope                   0.1.0                    py27_0  
backports                 1.0                      py27_0  
backports_abc             0.5                      py27_0  
bleach                    1.5.0                    py27_0  
configparser              3.5.0                    py27_0  
cycler                    0.10.0                   py27_0  
decorator                 4.0.11                   py27_0  
entrypoints               0.2.2                    py27_1  
enum34                    1.1.6                    py27_0  
freetype                  2.5.5                         2  
funcsigs                  1.0.2                    py27_0    conda-forge
functools32               3.2.3.2                  py27_0  
get_terminal_size         1.0.0                    py27_0  
html5lib                  0.999                    py27_0  
html5lib                  0.9999999                 <pip>
icu                       54.1                          0  
ipykernel                 4.6.1                    py27_0  
ipython                   5.3.0                    py27_0  
ipython_genutils          0.2.0                    py27_0  
jinja2                    2.9.6                    py27_0  
jsonschema                2.6.0                    py27_0  
jupyter_client            5.0.1                    py27_0  
jupyter_core              4.3.0                    py27_0  
libpng                    1.6.27                        0  
Markdown                  2.2.0                     <pip>
markupsafe                0.23                     py27_2  
matplotlib                2.0.2               np112py27_0  
mistune                   0.7.4                    py27_0  
mkl                       2017.0.1                      0  
mock                      2.0.0                    py27_0    conda-forge
nbconvert                 5.1.1                    py27_0  
nbformat                  4.3.0                    py27_0  
notebook                  5.0.0                    py27_0  
numpy                     1.13.0                    <pip>
numpy                     1.12.1                   py27_0  
openssl                   1.0.2l                        0  
pandas                    0.20.1              np112py27_0  
pandocfilters             1.4.1                    py27_0  
path.py                   10.3.1                   py27_0  
pathlib2                  2.2.1                    py27_0  
pbr                       3.0.1                    py27_0    conda-forge
pexpect                   4.2.1                    py27_0  
pickleshare               0.7.4                    py27_0  
pip                       9.0.1                    py27_1  
prompt_toolkit            1.0.14                   py27_0  
protobuf                  3.3.0                    py27_1    conda-forge
ptyprocess                0.5.1                    py27_0  
pygments                  2.2.0                    py27_0  
pyparsing                 2.1.4                    py27_0  
pyqt                      5.6.0                    py27_2  
python                    2.7.13                        0  
python-dateutil           2.6.0                    py27_0  
pytz                      2017.2                   py27_0  
pyzmq                     16.0.2                   py27_0  
qt                        5.6.2                         2  
readline                  6.2                           2  
scandir                   1.5                      py27_0  
scikit-learn              0.18.1              np112py27_1  
scipy                     0.19.0              np112py27_0  
seaborn                   0.7.1                    py27_0  
setuptools                36.0.1                    <pip>
setuptools                27.2.0                   py27_0  
simplegeneric             0.8.1                    py27_1  
singledispatch            3.4.0.3                  py27_0  
sip                       4.18                     py27_0  
six                       1.10.0                   py27_0  
sqlite                    3.13.0                        0  
ssl_match_hostname        3.4.0.2                  py27_1  
subprocess32              3.2.7                    py27_0  
tensorflow                1.1.0                    py27_0    conda-forge
tensorflow                1.2.0rc1                  <pip>
terminado                 0.6                      py27_0  
testpath                  0.3                      py27_0  
tk                        8.5.18                        0  
tornado                   4.5.1                    py27_0  
traitlets                 4.3.2                    py27_0  
wcwidth                   0.1.7                    py27_0  
Werkzeug                  0.12.2                    <pip>
werkzeug                  0.11.10                  py27_0    conda-forge
wheel                     0.29.0                   py27_0  
zlib                      1.2.8                         3  


## Data

All data required for the project is located in the 'data' folder. 

The original data was downloaded from the following url: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

The original training set is: 'kddcup.data_10_percet.gz', however the unzipped file was renamed to: KDD99Train.csv
The original testing set is: 'corrected.gz', however the unzipped file was renamed to: KDD99Test.csv

The files 'Attack Types.csv' and 'Field Names.csv' are adjusted versions of the following files, taken from the url mentioned above:

- training_attack_types
- kddcup.names

The adjustment was necessary to simplify preprocessing of the data. Further, not all attack types were included in the kddcup.names file. 

Attack types were taken from the following document: http://airccse.org/journal/nsa/0512nsa08.pdf


## Preprocessing

Preproccessing on the data was performed in the 'transform_data.ipynb' file. Most logic applied to the data is taken from methods in the module 'helpermModule.py'

Once preprocessing is performed the testing and training datasets are stored in the pickle file kdd99


## Training

The training and refinement process is extensively described in the capstone report.

Relevant files are:

-  basic_nn.ipynb
-  basic_rnn.ipynb
-  multicell_rnn.ipynb
-  multicell_rnn_dropout.ipynb
-  hyperparameter_tuning.ipynb
-  final_model.ipynb
-  helperModule.py

All jupyter notebooks and the helpermModule are extensively commented and should be straightforward to understand for an instructor.



Capstone Proposal link

https://review.udacity.com/?utm_medium=email&utm_campaign=reviewsapp-submission-reviewed&utm_source=blueshift&utm_content=reviewsapp-submission-reviewed&bsft_clkid=1b225dc7-dc99-41da-97ea-ce7480b4a1f1&bsft_uid=5baff027-5594-4dfa-92b1-884dea7262d1&bsft_mid=1ffd74fd-cd01-464f-b1f2-cae2d51f00ff&bsft_eid=6f154690-7543-4582-9be7-e397af208dbd&bsft_txnid=7caa7be8-365b-4e3d-8fb3-30cef4d33195#!/reviews/521705






