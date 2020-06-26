Problem Description
This is a code solution for a language modeling problem. It involves, trainning a language model using one of the novel of sherlock holmes and measuring its perplexity over and entirely different episode. 
Course of actions for the Trainning:

-	Having taken a closer look at the train and test corpus provided, it seem to be a complex problem. The complexities identified as:
o	Infrequent named entities, like character names, locations, numbers etc
o	Non-linear flow of text without any immediate and simpler context relation to the preceding text
o	Closed vocabulary, many tokens in the test-set unknown to the train-set
-	Approach: 
o	Out of the two possible options training the language model I opted for the deep-learning approach since any other approach like N-grams or bag of words would not have a considerable amount of corpus available to build. 
o	Idea behind applying deep-learning to the problem is that the writing style of author might have some patterns which can be interpreted in the vector representation of words. Training on these patterns might help to capture a model that can yield high probability on the words of test-set which might also follow a similar pattern. 
o	The out of vocabulary problem was dealt by setting a closed vocabulary and replacing infrequent words in training set with <unk> and train these words as normal words. On the other hand all the words from test-set not present in train-set are replaced with <unk> [1].  
o	As a choice of model-frame; input was chosen to be line by line sequences of tokens from sentences separated by punctuations (except commas). E.g. the sentence 

“i may want your help, and so may he.” 

can be represented as follow:

X										Y
I										may
I may									want
I may want								your
I may want your							help
I may want your help					and
I may want your help and				so
I may want your help and so				may
I may want your help and so may			he

o	   As a choice of network an LSTM with 64 units per cell was chosen, which takes pre trained word vectors (GloVe) as input, which are further trained on the fly.

Result:
-	The training was carried out with a manually chosen combination of 200 epochs with a batch-size of 128 samples. However the learning curve was extremely slow and the final accuracy was about 47%. Resulting in a near to infinite perplexity (overflow in python). I have figured out following factors for the slow learning. Improving upon which can improve the model performance.
o	Insufficient units in LSTM: the model might require a more richer representation of the hidden state hence increasing the units in LSTM layer or adding more additional layer might help.
o	Non-linear approach to the frame the input: a character or narrator level continuity in the utterances/passages and a larger input to accommodate that. Or passing an additional input,  a dense context vector using bag of word approach or the hidden state of the previous dialogs/passages.
o	The trainable feature of the input embeddings might also be a cause of the slow learning. Rather, the pre-trained embeddings should be used as it is, presuming that the global representation of vector might be well enough to represent the relation between words in the corpus. For words not in Glove another suitable approach can be thought of to get a dense vector e.g. bag of words or replacing these words with named entities.

Considering the fact that the problem of making a model predict words of Sir Arthur Conan Doyle, is a rather complex one, I have tried my best here to bring up an approach which serves good as a starting point.

PS: on a separate note below is a cool text passage that I could generate using “Sherlock” as the initiating word 


'sherlock holmes sat down beside the envelope and the boscombe pool is unk from a unk which lined it had been concerned in the unk of the unk bridegroom that they were in vain for the time of the unk of the rain of the unk face the smell of articles street in a row which lined it of a match and looked in the corner of the footpaths it was empty up to the corner of the ceiling the blue smoke curling up from the chimneys showed that was unk or torn from extreme languor to devouring energy and that the searching streatham circumstances private circumstances shut and his head double chink letter effort letter effort his truth lies straight major errand streets weather platform bow major inquiry at the hotel cosmopolitan wide pipes six beside england trouble effort vague straight effort effort bow arrest double bedded straight letter effort me then effort effort his eyes effort to request effort letter regular property contrast buried effort to interfere than the searching police stepping to private single tide free envelope age double admirably truth unk with a man who is effort weather age single roylott remained remarkably resemblance effort to recognise the unk which of poison threatened to affect the school weather elastic hurry as her presence broadened as a railway regular yellow action from his centre of unk carpet in a crackling regular footfall bow double match and windows age grounds weather and the unk unk double pluck effort me to himself in his own roof but he was hot clang effort effort letter regular drew with a request streets weather elastic was surely unk and america in sheer hat since the question of b regular yellow commission age effort from the unk effort of regular sandwiched effort to preserve it which contrast to contrast to day a telegram upon the sundial as to contrast double sweet major stick effort letter age double clump letter age garden streets weather elastic hurry weather elastic trouble effort effort effort letter effort effort over the unk of limbs effort effort effort letter to it which he effort effort effort to mother who knowledge effort me colour professional regular footfall professional straight to the imagination of limbs professional straight bow and his centre platform bow and bow tongue replied stepping bow streets bow and windows effort daring den effort form effort effort pawnbroker firm upon the shining finger professional double deduction which erected letter contrast buried a week power effort effort effort age contrast straight effort form effort of it for the unk title of the unk bridegroom cases which has been his unk just as to unk unk upon the previous night and of unk s clouds coroner school companion with a smile effort effort effort effort effort effort effort effort effort slipping skill and streets age trouble and all discoloured that he effort to affairs at all that buried in the dark riverside double deduction to be of tension going and'


# Requirements
## Data
### train: hound_train.txt
### test: hound_test.txt
### Gloobal word-vectors glove.6n.50d (not provided in the repository)

##Python
### numpy, nltk, pandas, collections, itertools, csv, re, tensorflow 
