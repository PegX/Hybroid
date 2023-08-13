import os
import shutil
import sys
#import urllib2
import re, operator

#filePath = "python_word_freqency.html"

# keys are words, vals are occurance frequency
freqlist={}
    
def single_file_word_frequency(path,file_):   
    inF = open(path+'/'+file_, "r")
    s=inF.read()
    inF.close()

    s=s.lower()

    wordlist = re.split(r'\W',s)
    for wd in wordlist:
        if wd in freqlist:
            freqlist[wd]=freqlist[wd]+1
        else:
            freqlist[wd]=1

def opcode_frequency(path_in,path_out):
    for root,dirs,files in os.walk(path_in):
        for file_ in files:
        #opcode_from_objdump("data/"+file_,"opcode/"+file_)
            single_file_word_frequency(path_in,file_)
    print("writing files")
    f_frequency = open(path_out+sys.argv[3]+".txt","w")
    for k,v in sorted(freqlist.items(), key=operator.itemgetter(1) ,reverse=True):
        #print(str(v) + "->" + k)
        f_frequency.write(k)
        f_frequency.write(' ')
        f_frequency.write(str(v))
        f_frequency.write('\n')

def move_file(path_in,path_out):
    print("Moving files")
    for dirName, subdirList, fileList in os.walk(path_in):
        for f in fileList:
            print(f)
            if f.endswith(".opcode"):
                shutil.move(path_in+f, path_out+"raw/"+f)



if __name__ == "__main__":
    print("starting")
    move_file(sys.argv[1],sys.argv[2])
    opcode_frequency(sys.argv[2]+"raw/",sys.argv[2])