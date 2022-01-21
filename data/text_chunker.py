#!/usr/bin/env python
# coding: utf-8


def chunker(input_data, N):
    #Split the input text into chunks, where each chunk contain N words
    input_words = input_data.split(' ')
    output = []
    
    cur_chunk = []
    count = 0
    
    for word in input_words:
        cur_chunk.append(word)
        count += 1
        
        if count == N:
            output.append(' '.join(cur_chunk))
            count, cur_chunk = 0, []
            
    output.append(' '.join(cur_chunk)) #This is to add the last remaining words if possible the length != N
    
    return output