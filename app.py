from flask import Flask,render_template,request
import pickle
import re
import nltk
import pickle
from clean import clean_text
from summary import gen_summary
app = Flask(__name__)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


######## MODELS #############
hotel=pickle.load(open('hotel.pkl','rb'))
spam=pickle.load(open('spam.pkl','rb'))
twitter=pickle.load(open('twitter30k.pkl','rb'))
movie=pickle.load(open('movie.pkl','rb'))
stock=pickle.load(open('stock.pkl','rb'))

######## Count Vectorizer ##########
hotelcv=pickle.load(open('hotelcv.pkl','rb'))
spamcv=pickle.load(open('spamcv.pkl','rb'))
twittercv=pickle.load(open('twitter30kcv.pkl','rb'))
moviecv=pickle.load(open('moviecv.pkl','rb'))
stockcv=pickle.load(open('stockcv.pkl','rb'))


####### GLOBAL CLEANING #######
def clean(new_review):
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = re.sub('<.*?>'," ",new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review] 
    return new_corpus



############# HOME ROUTE #########
@app.route('/')
def homepage():
    return render_template('index.html')


########## HOTEL REVIEW ANALYSIS ###########
@app.route('/hotel')
def Hotel():
    return render_template('hotel.html')
@app.route('/predicthotel',methods=['POST'])
def hotelPrediction():
    if request.method=='POST':
        new_review = request.form['review']
        corpus = clean(new_review)
        new_X_test = hotelcv.transform(corpus).toarray()
        pred = hotel.predict(new_X_test)
        return render_template('hotel_result.html',prediction=pred)

########### EMAIL SPAM DETECTION #######
@app.route('/spam')
def Spam():
    return render_template('spam.html')

@app.route('/predictspam',methods=['POST'])
def spamPrediction():
    mail = request.form['email']
    data = [mail]
    vect = spamcv.transform(data).toarray()
    pred = spam.predict(vect)
    return render_template('spam_result.html',prediction=pred)


################ TWITTER SENTIMENT ANALYSIS ############
@app.route('/twitter')
def Twitter():
    return render_template("twitter.html")

@app.route('/predicttwitter',methods=['POST'])
def twitterPrediction():
    if request.method=='POST':
        text = request.form['twitte']
        cleaned_text = clean_text(text)
        vectors =  twittercv.transform([cleaned_text]).toarray()
        pred = twitter.predict(vectors)
        return render_template('twitter_result.html',prediction=pred)

################# MOVIE REVIEW SENTIMENT #################
@app.route('/movie')
def Movie():
    return render_template("movie.html")

@app.route('/predictmovie',methods=['POST'])
def moviePrediction():
    if request.method=='POST':
        new_review = request.form['review']
        corpus = clean(new_review)
        new_X_test = moviecv.transform(corpus).toarray()
        pred = movie.predict(new_X_test)
        return render_template('movie_result.html',prediction=pred)

############## SUMMARY ####################

@app.route('/textsummary')
def home():
    return render_template("summary.html")
@app.route('/summary',methods=['POST'])
def data():
    url = request.form['url']
    num = request.form['count']
    num = int(num)
    summary = gen_summary(url,num)
    return render_template(f'summary.html',summary=f'{summary}')



################## STOCK HEADLINE ANALYSIS ##############↗↘
@app.route('/stock')
def stockanalysis():
    return render_template("stock.html")
@app.route('/predictstock',methods=['POST'])
def predictstock():
    headline = request.form['headline']
    headline = headline.lower()
    new_test = stockcv.transform([headline]).toarray()
    pred = stock.predict(new_test)[0]
    if pred>0:
        pred = "Stock Price Will Increase ↗↗"
    else:
        pred='Stock Price Will Stay The Same 〽'
    return render_template(f'stock.html',pred=pred)

if __name__ == "__main__":
    app.run(debug=True)    
