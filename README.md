https://github.com/Tarun-Elango/COMP-472-Mini-project-1.git

# COMP 472 Mini Project 1 

Instructions to run 

download all dependencies to local machine using pip or any package manager.
os-sys, sklearn, csv, json, gensim, pandas, numpy, matplotlib, nltk


## Question 1
cd Q1 <br />
run main.py in any python compiler or using command 'python main.py'. <br />
data.csv file generated, keep it here needed for next parts <br />


## Question 2
 
### For .ipynb files<br />

cd Q2<br />
cd into the respective folder (each model has its folder, along with Q 2.5. cd into any folder for the ipynb file )<br />
open Google colab on any browser, login with gmail account<br />
1. either go to upload section when promted to create a new notebook, or on the main page go to file and select upload notebook <br />
2. from the Q2 folder select the desired ipynb file to run and upload<br />
3. once opened, on the left side select the files icon(4th icon on the left panel) wait for it to connect<br />
4. right click inside the panel and select upload, and then upload the data.csv file that was genereted in Q1 (inside the Q1 folder), and do this for every notebook <br />
5. once uploaded, run the indiviudal code blocks=. <br />

### for .py files <br />
cd into Q2<br />
cd into respective folders(the desired model like naive bayes or neural network etc...)<br />
run the respective .py files in any python compiler or using the command 'python {python filename}.py'<br />


## Question 3 assuming its windows (for other OS, file location for gensim-data may or may not differ)
cd Q3<br />
1. run download.py (downloads the three pre trained model to local computer.)<br /> <br />
2. files downloaded in local machine at Users/{username}/gensim-data (cd into this folder)<br /> <br />
3. cd into local machine's Users/{username}/gensim-data/glove-twitter-25, unzip glove-twitter-25.gz(7 zip or breezip or any gzip opener), go into newly generated glove-twitter-25 file, copy glove-twitter-25.txt into 472_Assignment1_40084007/Q3/glove twitter/ <br /> <br />
4. cd back into local machine's Users/{username}/gensim-data/glove-wiki-gigaword-50, unzip glove-wiki-gigaword-50.gz(7 zip or breezip or any gzip opener), go into newly generated glove-wiki-gigaword-50 file, copy glove-wiki-gigaword-50.txt into 472_Assignment1_40084007/Q3/glove wiki 50/ <br /> <br />
5. cd back into local machine's Users/{username}/gensim-data/word2vec-google-news-300, unzip word2vec-google-news-30.gz(7 zip or breezip or any gzip opener), go into newly generated word2vec-google-news-30 file, copy word2vec-google-news-30.bin into 472_Assignment1_40084007/Q3/google word to vec/ <br /> <br />
6. Now cd back into Q3, select the desired model folder, and run the python files for the pre trained model(make sure the corresponding .bin or .txt files are present along the python file as mentioned above) on any python compiler or using the command 'python {python filename}.py'. 
