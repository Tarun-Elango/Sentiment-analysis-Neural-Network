import matplotlib.pyplot as plt
import json
import csv
import pandas as pd

# convert json to csv for ease of access
file = open('goemotions.json')
data = json.load(file)

file.close()
file = open('data.csv','w', encoding='utf-8')
csv_file = csv.writer(file)
csv_file.writerow(["post", "emotion", "sentiment"])
for item in data:
    csv_file.writerow(item)
file.close()


def print_emotion():
    fileEmo = pd.read_csv('data.csv')#open csv file
    figure, axEmo = plt.subplots()# create a subplot
    b= fileEmo.groupby('emotion').emotion.count() #group by emotion
    axEmo.pie(b, labels=b.keys(), autopct='%1.1f%%') #create a pie chart and show percentages
    plt.xlabel("emotion pie chart")
    plt.savefig("emotion.png")
    plt.show()

def print_sentiment():
    fileSent = pd.read_csv('data.csv')#open csv file
    figures, axSent = plt.subplots()# create a subplot
    a = fileSent.groupby('sentiment').sentiment.count() # group by sentiment
    axSent.pie(a, labels=a.keys(), autopct='%1.1f%%') #create a pie chart
    plt.xlabel("sentiment pie chart")
    plt.savefig("sentiment.png")
    plt.show()

print_emotion()
print_sentiment()


# reference
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html