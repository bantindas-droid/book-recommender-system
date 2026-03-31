import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

titles = [
    "Harry Potter",
    "Percy Jackson",
    "Lord of the Rings",
    "The Hobbit",
    "Game of Thrones",
    "Sherlock Holmes",
    "Pride and Prejudice",
    "The Fault in Our Stars",
    "Twilight",
    "The Notebook",
    "Me Before You",
    "It Ends With Us",
    "The Alchemist",
    "Diary of a Wimpy Kid",
    "The Hitchhiker's Guide to the Galaxy",
    "Good Omens",
    "The Rosie Project",
    "Can You Keep a Secret",
    "Love Story",
    "Romeo and Juliet",
    "Dune",
    "The Martian",
    "Gone Girl",
    "The Girl on the Train",
    "Verity",
    "Ugly Love",
    "The Love Hypothesis",
    "Beach Read",
    "Eleanor Oliphant Is Completely Fine",
    "Bossypants",
    "Five Point Someone",
    "Half Girlfriend",
    "The 3 Mistakes of My Life",
    "Train to Pakistan",
    "The White Tiger",
    "Wings of Fire",
    "Ignited Minds",
    "Rich Dad Poor Dad India",
    "You Can Win",
    "The God of Small Things"
]

authors = [
    "J.K. Rowling",
    "Rick Riordan",
    "J.R.R. Tolkien",
    "J.R.R. Tolkien",
    "George R.R. Martin",
    "Arthur Conan Doyle",
    "Jane Austen",
    "John Green",
    "Stephenie Meyer",
    "Nicholas Sparks",
    "Jojo Moyes",
    "Colleen Hoover",
    "Paulo Coelho",
    "Jeff Kinney",
    "Douglas Adams",
    "Neil Gaiman",
    "Graeme Simsion",
    "Sophie Kinsella",
    "Erich Segal",
    "William Shakespeare",
    "Frank Herbert",
    "Andy Weir",
    "Gillian Flynn",
    "Paula Hawkins",
    "Colleen Hoover",
    "Colleen Hoover",
    "Ali Hazelwood",
    "Emily Henry",
    "Gail Honeyman",
    "Tina Fey",
    "Chetan Bhagat",
    "Chetan Bhagat",
    "Chetan Bhagat",
    "Khushwant Singh",
    "Aravind Adiga",
    "A.P.J. Abdul Kalam",
    "A.P.J. Abdul Kalam",
    "Robert Kiyosaki",
    "Shiv Khera",
    "Arundhati Roy"
]

genres = [
    "Fantasy",
    "Fantasy",
    "Fantasy",
    "Fantasy",
    "Fantasy",
    "Mystery",
    "Romance",
    "Romance",
    "Romance",
    "Romance",
    "Romance",
    "Romance",
    "Fiction",
    "Comedy",
    "Comedy",
    "Comedy",
    "Comedy",
    "Comedy",
    "Romance",
    "Romance",
    "Science Fiction",
    "Science Fiction",
    "Thriller",
    "Thriller",
    "Thriller",
    "Romance",
    "Romance",
    "Romance",
    "Fiction",
    "Comedy",
    "Fiction",
    "Romance",
    "Fiction",
    "Historical",
    "Fiction",
    "Biography",
    "Motivation",
    "Finance",
    "Motivation",
    "Fiction"
]

descriptions = [
    "magic wizard school adventure",
    "demigod greek mythology adventure",
    "epic fantasy ring journey",
    "dragon adventure fantasy quest",
    "kingdom war politics dragons",
    "detective mystery investigation",
    "classic love story society manners",
    "teen love cancer emotional story",
    "vampire love supernatural romance",
    "deep emotional love story",
    "romantic tragic relationship",
    "modern love emotional drama",
    "philosophical journey self discovery",
    "funny school life comedy diary",
    "science fiction comedy space adventure",
    "humor angels demons comedy",
    "romantic comedy awkward relationship",
    "funny secret romance life",
    "tragic young love story",
    "classic tragic romance",
    "desert planet politics survival",
    "astronaut survival mars science",
    "psychological thriller mystery crime",
    "missing girl mystery thriller",
    "dark romance psychological thriller",
    "emotional romance heartbreak story",
    "science romance academic love",
    "romantic writers love story",
    "lonely woman life journey",
    "comedy memoir humor life",
    "college life friendship engineering story",
    "india love story relationship drama",
    "business cricket life story india",
    "partition india pakistan emotional story",
    "poor boy success india story",
    "scientist life inspiration india",
    "youth motivation dreams india",
    "money finance mindset india",
    "self help success motivation",
    "family drama kerala india story"
]

df = pd.DataFrame({
    "title": titles,
    "author": authors,
    "genre": genres,
    "description": descriptions
})

df["features"] = df["author"] + " " + df["genre"] + " " + df["description"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["features"])

similarity = cosine_similarity(tfidf_matrix)

def recommend(book_name):
    if book_name not in df["title"].values:
        print("Book not found")
        return

    index = df[df["title"] == book_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:8]

    print("\nRecommended Books:\n")
    for i in scores:
        print(df.iloc[i[0]]["title"])

print("Book Recommender System")

while True:
    user_input = input("\nEnter a book name (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Goodbye")
        break

    recommend(user_input)