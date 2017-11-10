import nltk
#nltk.download("movie_reviews", download_dir = 'C:\\Users\\yanlu\\Documents\\Data Science\\Week-8-NLP-Databases')

from nltk.corpus import movie_reviews
print(len(movie_reviews.fileids()))
print(movie_reviews.fileids()[:5])
print(movie_reviews.fileids()[-5:])
#fileids can also filter the available files based on their category,
# which is the name of the subfolders they are located in.
negative_fileids = movie_reviews.fileids('neg')
positive_fileids = movie_reviews.fileids('pos')

len(negative_fileids)
len(positive_fileids)

print(movie_reviews.raw(fileids=positive_fileids[0]))

#Tokenize text in words
romeo_text = """Why then, O brawling love! O loving hate!
O any thing, of nothing first create!
O heavy lightness, serious vanity,
Misshapen chaos of well-seeming forms,
Feather of lead, bright smoke, cold fire, sick health,
Still-waking sleep, that is not what it is!
This love feel I, that feel no love in this."""
romeo_text.split()
#nltk.download("punkt", download_dir = 'C:\\Users\\yanlu\\Documents\\Data Science\\Week-8-NLP-Databases')
# word_tokenize function to properly tokenize this text
romeo_words = nltk.word_tokenize(romeo_text)
#print(romeo_text)

movie_reviews.words(fileids=positive_fileids[0])