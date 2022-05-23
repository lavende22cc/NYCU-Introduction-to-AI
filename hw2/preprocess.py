from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import nltk,string
from nltk.stem import SnowballStemmer


 

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    # Begin your code (Part 0)
    s = text
    s = s.lower()                #將所有字母轉成小寫
    s = s.replace('<br />' , '')    #將html換行字元去除
    s = remove_stopwords(s)         #使用nltk lib中的function移除stopwords
    words = s.split()

    #stemming
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]

    # Remove punctuations
    words = [word for word in words if word not in string.punctuation]

    # get result
    preprocessed_text = ' '.join([word for word in words])

    # End your code

    return preprocessed_text


if __name__ == "__main__":
    s = "If you have not installed NumPy or SciPy yet, <br /><br /> you can also install these using conda or pip."
    print('Before : \n' , s);
    print('After : \n' , preprocessing_function(s))
