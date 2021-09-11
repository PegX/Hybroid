# Hybroid
This repo is the implementation of our ISC2021 paper. 

Hybroid: Toward Android Malware Detectionand Categorization with Program Code and Network Traffic

In this paper, we present Hybroid, a hybrid Android malware detection and categorization solution that utilizes program code structures  as  static  behavioral  features  and  network  traffic  as  dynamic  behavioral features for detection (binary classification) and categorization(multi-label classification). For static analysis, we introduce a natural-language-processing-inspired technique based on function call graph embeddings and design a graph-neural-network-based approach to convert the whole graph structure of an Android app to a vector. For dynamic analysis, we extract network flow features from the raw network traffic by  capturing  each  application’s  network  flow.  Finally, Hybroid utilizes the network flow features combined with the graphs’ vectors to detect and categorize the malware. Our solution demonstrates 97.0% accuracy on average for malware detection and 94.0% accuracy for malware categorization. Also, we report remarkable results in different performance metrics such as F1-score, precision, recall, and AUC.

## Dependency
Androguard
Tensorflow

## Implementation
### Program Code feature extraction
Hybroid extracts program code feature from Android apps directly. First of all, we extract bytecode(instructions, control flow graph, function call graph and so on) from the DEX file of Android application, and then we use the techqniues from the natural language processing(NLP) to cnovert the instruction(opcode) inside of bytecode to vector by using the variant of word2vec method. After we get the instruction2vec, we assemble these instrcution2vec to function2vect. Last but not least, we also conver the whole function call graph (control flow graph) to an vector, which stands for the whole feature of an Android application. 

The implementation can be found under the code_graph_embedding folder. 
### Network Traffic feature extraction
