from flask import Flask, render_template,request
import pickle
import numpy as np 

popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
book = pickle.load(open('book.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

app = Flask(__name__)

@app.route("/") 
def index() :
    return render_template('index.html',
                        book_name = list(popular_df['Book-Title'].values),
                        author = list(popular_df['Book-Author'].values),
                        image = list(popular_df['Image-URL-M'].values),
                        votes = list(popular_df['num_ratings'].values),
                        rating = list(popular_df['avg_ratings'].values),
                        )

@app.route('/recommend')
def recommand_ui() : 
    return render_template('recommand.html')

@app.route('/recommend_books',methods=['post'])
def recommand() :
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0] 
    distance = similarity[index]
    similarItem = sorted(list(enumerate(similarity[index])),key=lambda x:x[1],reverse=True)[1:10]
    data = []
    for i in similarItem: 
        item = []
        # print(pt.index[i[0]])
        temp_df = book[book['Book-Title'] == pt.index[i[0]]] 
        item.append(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.append(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.append(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)
        # print(data)
    return render_template('recommand.html',data=data)
    # return str(data[:][2])

if __name__ == '__main__' :
    app.run(debug=True)