from flask import Flask, render_template, request
from pathlib import Path
from difflib import get_close_matches
import pickle
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'Model'
MAX_RECOMMENDATIONS = 10


def _load_csv(filename):
    csv_path = MODEL_DIR / filename
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding='latin-1')


def _compute_similarity(matrix):
    # Use cosine similarity without an external dependency.
    values = matrix.astype(float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1e-9, norms)
    return (values @ values.T) / (safe_norms @ safe_norms.T)


def _normalize_title(value):
    if pd.isna(value):
        return ''
    normalized = ' '.join(str(value).strip().lower().split())
    return normalized


def _build_artifacts_from_csv():
    books = _load_csv('Books.csv')
    ratings = _load_csv('Ratings.csv')

    ratings_with_name = ratings.merge(books, on='ISBN')

    num_rating_df = (
        ratings_with_name.groupby('Book-Title')
        .count()['Book-Rating']
        .reset_index()
        .rename(columns={'Book-Rating': 'num_ratings'})
    )

    avg_rating_df = (
        ratings_with_name.groupby('Book-Title')
        .mean(numeric_only=True)['Book-Rating']
        .reset_index()
        .rename(columns={'Book-Rating': 'avg_ratings'})
    )

    popular_df_local = num_rating_df.merge(avg_rating_df, on='Book-Title')
    popular_df_local = popular_df_local[popular_df_local['num_ratings'] >= 250]
    popular_df_local = (
        popular_df_local.sort_values('avg_ratings', ascending=False)
        .head(50)
        .merge(books, on='Book-Title')
        .drop_duplicates('Book-Title')
    )

    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    active_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]

    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_rating = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

    pt_local = final_rating.pivot_table(
        index='Book-Title', columns='User-ID', values='Book-Rating'
    ).fillna(0)
    similarity_local = _compute_similarity(pt_local.values)

    book_local = books
    return popular_df_local, pt_local, book_local, similarity_local


def _build_title_stats(book_df):
    ratings = _load_csv('Ratings.csv')
    merged = ratings.merge(book_df, on='ISBN')

    stats = (
        merged.groupby('Book-Title')['Book-Rating']
        .agg(['count', 'mean'])
        .reset_index()
        .rename(columns={'count': 'num_ratings', 'mean': 'avg_ratings'})
    )

    max_count = max(float(stats['num_ratings'].max()), 1.0)
    stats['popularity_norm'] = stats['num_ratings'] / max_count
    stats['norm_title'] = stats['Book-Title'].map(_normalize_title)
    return stats


def _load_or_build_artifacts():
    popular_path = BASE_DIR / 'popular.pkl'
    pt_path = BASE_DIR / 'pt.pkl'
    book_path = BASE_DIR / 'book.pkl'
    similarity_path = BASE_DIR / 'similarity.pkl'

    if all(path.exists() for path in [popular_path, pt_path, book_path, similarity_path]):
        popular_df_local = pickle.load(open(popular_path, 'rb'))
        pt_local = pickle.load(open(pt_path, 'rb'))
        book_local = pickle.load(open(book_path, 'rb'))
        similarity_local = pickle.load(open(similarity_path, 'rb'))
        return popular_df_local, pt_local, book_local, similarity_local

    popular_df_local, pt_local, book_local, similarity_local = _build_artifacts_from_csv()

    with open(popular_path, 'wb') as f:
        pickle.dump(popular_df_local, f)
    with open(pt_path, 'wb') as f:
        pickle.dump(pt_local, f)
    with open(book_path, 'wb') as f:
        pickle.dump(book_local, f)
    with open(similarity_path, 'wb') as f:
        pickle.dump(similarity_local, f)

    return popular_df_local, pt_local, book_local, similarity_local


popular_df, pt, book, similarity = _load_or_build_artifacts()
title_stats = _build_title_stats(book)

title_catalog = (
    pd.DataFrame({'Book-Title': pt.index})
    .assign(norm_title=lambda x: x['Book-Title'].map(_normalize_title))
)

title_to_original = {
    row.norm_title: row['Book-Title']
    for _, row in title_catalog.iterrows()
}

title_stats_by_name = title_stats.set_index('Book-Title')
suggested_titles = sorted(pt.index.tolist())


def _resolve_title(user_input):
    cleaned = _normalize_title(user_input)
    if not cleaned:
        return None, 'Please enter a book title.'

    if cleaned in title_to_original:
        return title_to_original[cleaned], None

    contains_matches = title_catalog[
        title_catalog['norm_title'].str.contains(cleaned, regex=False)
    ]
    if not contains_matches.empty:
        best = contains_matches.iloc[0]['Book-Title']
        return best, f'Using closest match: "{best}"'

    closest = get_close_matches(cleaned, title_catalog['norm_title'].tolist(), n=1, cutoff=0.6)
    if closest:
        best = title_to_original[closest[0]]
        return best, f'Using closest match: "{best}"'

    return None, 'No close title found. Try another book name.'


def _build_recommendations(selected_title):
    selected_index = np.where(pt.index == selected_title)[0][0]
    candidates = sorted(
        list(enumerate(similarity[selected_index])),
        key=lambda x: x[1],
        reverse=True,
    )[1:80]

    scored = []
    for idx, sim_score in candidates:
        candidate_title = pt.index[idx]
        stats_row = title_stats_by_name.loc[candidate_title] if candidate_title in title_stats_by_name.index else None
        popularity_bonus = float(stats_row['popularity_norm']) if stats_row is not None else 0.0
        weighted_score = 0.78 * float(sim_score) + 0.22 * popularity_bonus

        temp_df = book[book['Book-Title'] == candidate_title].drop_duplicates('Book-Title')
        if temp_df.empty:
            continue

        cover = temp_df.iloc[0].get('Image-URL-M', '')
        author = temp_df.iloc[0].get('Book-Author', 'Unknown')
        scored.append(
            {
                'title': candidate_title,
                'author': author,
                'image': cover,
                'similarity': round(float(sim_score) * 100, 1),
                'score': weighted_score,
            }
        )

    final_ranked = sorted(scored, key=lambda x: x['score'], reverse=True)[:MAX_RECOMMENDATIONS]
    return final_ranked

app = Flask(__name__)

@app.route("/")
def index():
    return render_template(
        'index.html',
        book_name=list(popular_df['Book-Title'].values),
        author=list(popular_df['Book-Author'].values),
        image=list(popular_df['Image-URL-M'].values),
        votes=list(popular_df['num_ratings'].values),
        rating=list(popular_df['avg_ratings'].values),
    )

@app.route('/recommend')
def recommand_ui():
    return render_template('recommand.html', suggestions=suggested_titles)

@app.route('/recommend_books', methods=['post'])
def recommand():
    user_input = request.form.get('user_input', '')
    selected_title, info_message = _resolve_title(user_input)

    if not selected_title:
        return render_template(
            'recommand.html',
            data=[],
            selected_title=user_input,
            message=info_message,
            suggestions=suggested_titles,
        )

    data = _build_recommendations(selected_title)
    return render_template(
        'recommand.html',
        data=data,
        selected_title=selected_title,
        message=info_message,
        suggestions=suggested_titles,
    )

if __name__ == '__main__':
    app.run(debug=True)