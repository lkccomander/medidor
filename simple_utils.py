# simple_utils.py - A tiny utility library

def reverse_string(text):
    """Reverses the characters in a string."""
    return text[::-1]

def count_words(sentence):
    """
    Count the number of words in a sentence by splitting on whitespace.
    
    Counts whitespace-separated tokens in `sentence`. Consecutive whitespace is treated as a single separator and leading/trailing whitespace is ignored.
    
    Parameters:
        sentence (str): Input text to count words in.
    
    Returns:
        int: Number of words in the input sentence.
    """
    return len(sentence.split())

def celsius_to_fahrenheit(celsius):
    """
    Convert a Celsius temperature to Fahrenheit.
    
    Parameters:
        celsius (int | float): Temperature in degrees Celsius.
    
    Returns:
        float: Temperature in degrees Fahrenheit.
    """
    return (celsius * 9/5) + 32