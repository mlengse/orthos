**August 1983 Report No. STAN-CS-83-977** 

Word Hy-phen-a-tion by Com-put«er 

**Franklin Mark Liang** 

Deparlment of Computer Science 

**Stanford University** 

**Stanford, CA 94305**  
**WORD HY-PHEN-A-TION BY COM-PUT-ER** 

**Franklin Mark Liang** 

**Department of Computer Science** 

**Stanford University** 

**Stanford, California 94305** 

**• Abstract**   
***\* •*** **. This thesis describes research leading to an improved word hyphenation algo rithm for the TjrjX82 typesetting system. Hyphenation is viewed primarily as a data compression problem, where we are given a dictionary of words with allowable divi sion points, and try to devise methods that take advantage of the large amount of redundancy present.** 

**The new hyphenation algorithm is based on the idea of hyphenating and in hibiting patterns. These are simply strings of letters that, when they match hi a word, give us information about hyphenation at some point in the pattern. For example, '-tion' and \*c-c' are good hyphenating patterns. An important feature of this method is that a suitable set of patterns can be extracted automatical\!/ from the dictionary.** 

**In order to represent the set of patterns in a compart form that is also reasonably efficient for searching, the author has developed a new data structure called a packed trie. This data structure allows the very fast search times characteristic of indexed tries, but in many cases it entirely eliminates the wasted space for null links usually present in such tries. We demonstrate the versatility and practical advantages of this data structure by using a variant of it as the critical component of the program that generates the patterns from the dictionary.** 

**The resulting hyphenation algorithm uses about 4500 patterns that compile into a packed trie occupying 25K bytes of storage. These patterns find 89% of the hyphens in a pocket dictionary word list, with essentially no error. By comparison, the uncompressed dictionary occupies over 500K bytes.** 

***This research was supported in part by the National Science Foundation under grants IST-8B 01926 and MSC-8S-00984, and by the System Development Foundation. 'TgK' is a trademark of the American Mathematical Society.***  
**`WORD HY-PHEN-A-TION`** 

**`BY COM-PUT-ER`** 

**A DISSERTATION** 

**SUBMITTED TO THE DEPARTMENT OF COMPUTER SCIENCE f** 

**AND THE COMMITTEE ON GRADUATE STUDIES OF STANFORD UNIVERSITY** 

**W PARTIAL** FULFILLMENT **OF** THE **REQUIREMENTS** FOR THE DEGREE OF 

DOCTOR OF PHILOSOPHY 

by 

Franklin **Mark Liang** 

**June 1983** 

**ii**  
***©*** **Copyright 1983 by** 

**Franklin Mark Liang iii**  
Acknowledgments 

I am greatly indebted to my adviser, Donald Knuth, **for creating the research** environment that made this work possible. When I began **work on the I^jX project** as a summer job, I would not have predicted that computer typesetting would become such an active area of computer science research. **Prof.** Knuth's foresight was to recognize that there were a number of fascinating problems in the field waiting to be explored, and his pioneering efforts have stimulated many others **to** think about these problems. 

I am also grateful to the Stanford Computer Science Department **for** providing the facilities and the community that have formed the major part of my life for **the** past several yean. 

I thank my readers, Luis Trabb Pardo and John Gill, **as well as Leo Guibas** who served on my orals committee on short notice. 

In addition, thanks to David Fuchs and Tom **Pressburger for helpful advice** and encouragement. 

Finally, this thesis is dedicated to my parents, for whom **the** experience **of** pursuing a graduate degree has bee.i perhaps even more traumatic than it was for myself. 

**IV**  
Table of contents 

**Introduction 1 Examples . 2 T^X and hyphenation 3 Time magazine algorithm 4 Patterns 5 Overview of thesis 7** 

**The dictionary problem 8 Data structures 0 Superimposed coding . 10 Tries 11 Packed tries 15 Suffix compression . . . . . 16 Derived forms 18 Spelling checkers 10 Related work 21** 

**Hyphenation 23 Finite-state machines with output . 23 Minimization with don't cares . 24 Pattern matching 26** 

**Pattern generation 29 Heuristics 30 Collecting pattern statistics 31 Dynamic packed tries . 32 Experimental results 34 Examples .3 7** 

**History and Conclusion 39** 

**Appendix .... . . 45 The PATGEW program . , .4 5 List of patterns • ... . 74** 

**References .... . . . . . . . . . . . 83**  
*Chapter 1* 

**Introduction** 

*\** The work described in this thesis was inspired by the need for a word hyphen ation routine as part «f Don Knuth's T^jX typesetting system \[1\]. This system was initially designed in order to typeset Prof. Knuth's seven-volume series of books, *The Art of Computer Programming,* when he became dissatisfied with the qual ity of computer typesetting done by his publisher. Since Prof. Knuth's books were to be a definitive treatise on computer science, he could not bear to see his schol arly work presented in an inferior manner, when the degradation was entirely due to the fact that the material had been typeset by a computer\! 

Since then, TjrjX (also known as Tau Epsilon Chi, a system for technical text) has gained wide popularity, and it is being adopted by the American Mathematical Society, the world's largest publisher of mathematical literature, for use in its jour nals. TjjjX is distinctive among other systems for word processing/document prepa ration in its emphasis on the highest quality output, especially for technical mate rial. 

One necessary component of the system is a computer-based algorithm for hy phenating English words. This is part of the paragraph justification routine, and it is intended to eliminate the need for the user to specify word division points explic itly when they are necessary for good paragraph layout. Hyphenation occurs rela tively infrequently in most book-format printing, but it becomes rather critical in narrow-column formats such as newspaper printing. Insufficient attention paid to this aspect of layout results in large expanses of unsightly white space, or (even worse) in words split at inappropriate points, e.g. new-spaper. 

Hyphenation algorithms for existing typesetting systems are usually either rule based or dictionary-based. Rule-based algorithms rely on a set of division rules such as given for English in the preface of Webster's Un;ibridged Dictionary \[2\]. These in clude recognition of common prefixes and suffixes, splitting between double conso nants, and other more specialized rules. Some of the "rules" are not particularly  
**2 INTRODUCTION**   
**•** 

amenable to computer implementation; e.g. "split between **the elements of a** com pound word". Rule-based schemes are inevitably subject to error, and they rarely cover all possible cases. In addition, the task of finding a suitable set of rules in the first place can be a difficult and lengthy project. 

Dictionary-based routines sim'ply store an entire word list along with the allow able division points. The obvious disadvantage of this method is the excessive stor age required, as well as the slowing down of the justification process when the hy phenation routine needs to access a part of the dictionary on secondary store. 

Examples 

To demonstrate the importance of hyphenation, consider Figure 1, which **shows** a paragraph set in three different ways by T£p(. The first example uses TjjjX's nor mal paragraph justification parameters, but with the hyphenation routine turned off. Because the line width in this example is rather narrow, TjjX is unable to find an acceptable way of justifying the paragraph, resulting in the phenomenon known as an "overfull box\*. 

One way to fix this problem is to increase the "stretchability" of the spaces be tween words, as shown in the second example. (TjjX users: This was done by in creasing the stretch component of spaceship to . 5em.) The right margin is now straight, as desired, but the overall spacing is somewhat loose. 

In the third example, the hyphenation routine is turned on, and everything is beautiful. 

**In olden time\* when wishing till helped one, there lived a king whose daughters were all beautifi I, jut the youngest WM «O beautiful hat the sun itself, which has seer 10 much, was astonished wheneve r I ihone in her face. Close by he king's custlc lay a great dark orcat, and under nn old lime-trcr n tlir Forest « » a well, and when the Hay wns very warm, the king\*: child went out into the forest and sat down by the side of the tool 'nuntain, and when shr was borcc ihe took a golden ball, and threw I up on high and caught it, and this hall was her favorite plaything.**   
**In olden limes when wishing still helped one, there lived a king whose daughters were all beautiful, but the youngest was so beautiful that the sun itself, which has seen so much, was astonished whenever it shone in her face. Close by the king's castle lay a great dark forest, and under *t \\* old lime tree in the forest was a well, and when the day was very warm, the king's child went out into the forest and sat down by the side of the cool fountain, and when she was bored she took a golden ball, and threw it up on high and caught it, and this ball was her favorite plaything.**   
**In olden times'when wish ing still helped one, there lived a king whose daughters were all beautiful, but the youngest waa so beautiful that the tun itself, which has seen so much, was as tonished whenever it shone in her face. Close by the king's castle lay a great dark forest, and un der an old lime-tree in the forest was a well, and when the day waa very warm, the king's child went out into the forest and sat down by the side of the cool fountain, and when she was bored she look** 

**\*. golden ball, and threw it up on high and cnuplit it, and this ball was her favorite plaything.** 

***Figure 1\. A typical paragraph with and without hyphenation.***  
`INTRODUCTION . S` 

**`sel-fadjotnt as-so-ciate as-so-cl-ate`** 

**`Pit-tsburgh prog-ress pro-gresa`** 

`clcarin-ghouee rec-ord re-cord` 

**`fun-draising a-rith me-tic ar-ith-met-ic ho-meowners eve-ning even-ing`** 

**`playw-right pe-ri-od-ic per-i-o-dic`** 

`algori-thm` 

**`walkth-rough in-de-pen-dent in-de-jend-ent Re-agan tri-bune`** `trib-une` 

*Figure 2\. Difficult hyphenations.* 

However, life is not always so simple. Figure 2 shows that hyphenation can be difficult. The first column shows erroneous hyphenations made by various typeset ting systems (which shall remain nameless). The next group of examples are words that hyphenate differently depending on how they are used. This happens most commonly with words that can serve as both nouns and verbs. The last two ex amples show that different dictionaries do not always agree on hyphenation (in this case Webster's vs. American Heritage). 

TjrjX and hyphenation 

The original TgX hyphenation algorithm was designed by Prof. Knuth and the author in the summer of 1977\. It is essentially a rule-based algorithm, with three main types of rules: (1) suffix removal, (2) prefix removal, and (3) vowel consonant-consonant-vowel (veev) breaking. The latter rule states that when the pattern 'vowel-consonant-consonant-vowcl' appears in a word, we can in most cases split between the consonants. There are also many special case rules; for example, "break vowel-q" or "break after ck". Finally a small exception dictionary (about 300 words) is used to handle particularly objectionable errors made by the above rules, and to hyphenate certain common words (e.g. pro-gram) that are not split by the rules. The complete algorithm is described in Appendix H of the old TjjX man ual. 

In practice, the above algorithm has served quite well. Although it does not find all possible division points in a word, it very rarely makes an error. Tests on a pocket dictionary word list indicate that about 40% of the allowable hyphen points are found, with *1%* error (relative to the total number of hyphen points). The al gorithm requires 4K 36-bit words of code, including the exception dictionary.  
**4 INTRODUCTION** 

The goal of the present research was to develop a better hyphenation algo rithm. By "better" we mean finding more hyphens, with little or no error, and us ing as little additional space as possible. Recall that one way to perform hyphen ation is to simply store the entire dictionary. Thus we can view our task as a data compression problem. Since there is a good deal of redundancy in English, we can hope for substantial improvement over the straightforward representation. 

Another goal was to automate the design of the algorithm as much as pos sible. The original T|rjX algorithm was developed mostly by hand, with a good deal of trial and etror. Extending such a rule-based scheme to find the remain ing hyphens seems very difficult. Furthermore such an effort must be repeated for each new language. The former approach can be a problem even for English, be cause pronunciation (and thus hyphenation) tends to change over time, and be cause different types of publication may call for different sets of admissible hy phens. *,* 

Time magazine algorithm 

A number of approaches were considered, including methods that have been dis cussed in the literature or implemented in existing typesetting systems. One of the methods studied was the so-called Time magazine algorithm, which is table-based rather than rule-based. 

The idea is to look at four letters surrounding each possible ' Breakpoint, namely two letters preceding and two letters following the given point. However we do not want to store a table of 264 \= 456,976 entries representing all possible four-letter combinations. (In practice only about 15% of these four-letter combinations actu ally occur in English words, but it is not immediately obvious how to take advan tage of this.) 

Instead, the method uses three tables of size 262, corresponding to the two let ters preceding, surrounding, and following a potential hyphen point. That is, if the letter pattern wx-yz occurs in a word, we look up three values correspond ing to the letter pairs wx, xy, and yz, and use these values to determine if we can split the pattern. 

What should the three tables contain? In the T:\\ne algorithm the table values were the probabilities that a hyphen could occur after, between, or before two given letters, respectively. The probability that the pattern wx-yz can be split is then es timated as the product of these three values (as if the probabilities were indepen dent, which they aren't). Finally the estimated value is compared against a thresh old to determine hyphenation. Figure 3 shows an example of hyphenation proba bilities computed by this method.  
**INTRODUCTION** 

**1,1 I**   
**`SUPERCALIFRAGIIISTICEIPIALIDOCIOU S`** 

***Figure S. Hyphenation probabilities.*** 

The advantage of this table-based approach is that the tables can be gen erated automatically from the dictionary. However, some experiments with the method yielded discouraging results. One estimate is 40% of the hyphens found, with 8% error. Thus a large exception dictionary would be required for good per formance. 

The reason for the limited performance of the above scheme is that just four let ters of context surrounding the potential break point are not enough in many cases. In an extreme example, we might have to look as many as 10 letters ahead in **or der** to determine hyphenation, e.g. dem-on-stra-tion vs. de-mpn-stra-tive. 

So a more powerful method is needed. • **Patterns**   
**A** good deal of experimentation led the author to a more powerful method based on the idea of hyphenation *patterns.* These ore simply strings of letters that, when they match in a word, will tell us how to hyphenate at some point in the pat tern. For example, the pattern 'tion' might tell us that we can hyphenate be fore the V. Or when the pattern 'cc' appears in a word, we can usually hy phenate between the c's. Here arc some more examples of good hyphenating pat terns: 

**.in-d .in-s** .in-t .un-d b-s \-cia con-s con-t **e-ly** er-1 er-ra ex- **\-ful it-t** i-ty \-lees 1-ly \-ment **n-co \-ness n-f n-1 n-ei n-v** om-m **\-sion** 8-ly s-nos ti-c a x-p 

(The character '. ' matches the beginning or end of a word.)  
**6 INTRODUCTION** 

Patterns have many advantages. They arc a general form of "hyphenation rule" that can include prefix, suffix, and other rules as special cases. Patterns can even de scribe an exception dictionary, namely by using entire words as patterns. (Actu ally, patterns are often more concise than an exception dictionary because a sin gle pattern can handle several variant forms of a word; e.g. pro-gram, pro-grams, and pro-grammed.) 

More importantly, the pattern matching approach has proven very effective. An ft 

appropriate set of patterns captures very concisely the information needed to per form hyphenation. Yet the pattern rules are of simple enough form that they can be generated automatically from the dictionary. 

When looking for good hyphenating patterns, we soon discover that almost all of them have some exceptions. Although \-tion is a very "safe" pattern, it fails on the word cat-ion. Most other cases are less clear-cut; for example, the common pat tern n-t can be hyphenated about SO percent of the time. It definitely seems worth while to use such patterns, provided that we can deal with the exceptions in some manner. 

After chooskg a set of hyphenating patterns, we may end up with thousands of exceptions. Theie could be listed in an exception dictionary, but we soon no tice there are many similarities among the exceptions. For example, in the orig inal T\]jjX algorithm we found that the vowcl-consonant-consonant-vowel rule re sulted in hundreds \<\>f errors of the form X-Yer or X-Yers, for certain consonant pairs XY, so we put in a new rule to prevent those errors. 

Thus, there may be "rules" that can handle large classes of exceptions. To take advantage of this, patterns come to the rescue again; but this time they are *inhibit-\* ivg* patterns, because they show where hyphens should *not* be placed. Some good ex imples of inhibiting patterns are: b=ly (don't break between b and ly), bs=, \=cing, io=n, i«tin, «l8, nn», nsst, n=ted, \=pt, ti=al, \=tly, «ts, and tt«. 

As it turns out, this approach is worth pursuing further. That is, after ap plying hyphenating and inhibiting patterns as discussed above, we might have an other »e( of hyphenating patterns, then another set of inhibiting patterns, and BO on. We can think of each level of patterns as being "exceptions to the ex ceptions" of the previous level. The current Tj}X82 algorithm uses five alternat ing levels of hyphenating and inhibiting patterns. The reasons for this will be ex plained in Chapter 4\. 

The idea of patterns is the basis of the new TJJX hyphenation algorithm, and it was the inspiration for much of the intermediate investigation, that will be de scribed.  
**INTRODUCTION 7** 

Overview **of** thesis 

In developing the pattern scheme, two main questions arose: (1) How can we represent the set of hyphenation patterns in a compact form that is also reason ably efficient for searching? (2) Given a hyphenated word list, how can we gener ate a suitable set of patterns? 

To solve these problems, the author has developed a new data structure called a *parked trie.* This data structure allows the very fast search times characteris tic of indexed tries, but in many cases it entirely eliminates the wasted space **for** null links usually present in such tries. 

We will demonstrate the versatility and practical advantages of this data struc ture \*y using it not only to represent the hyphenation patterns in the final algo rithm, but also *d'j* the critical component of the program that generates the pat terns from the dictionary. Packed tries have many other potential applications, in cluding identifier lookup, spelling checking, and lexicographic sorting. 

Chapter 2 considers the simpler problem of recognizing, rather than hyphenat ing, a set of words such as a dictionary, and uses this problem to motivate and ex plain the advantages of the packed trie data, structure. We also point out the close re lationship between tries and finite-state machines. 

Chapter 3 discusses ways of applying these ideas to hyphenation. After con sidering various approaches, including minimization with don't cares, we return to the idea of patterns. 

Chapter 4 discusses the heuristic method used to select patterns, introduces dy namic packed tries, and describes some experiments with the pattern generation pro-\* gram. 

Chapter 5 gives a brief history, and mentions ideas for future research. Finally, the appendix contains the WEB \[3\] listing of the portable pattern gen eration program PATGEN, as well as the set of patterns currently used by Tj£X82. 

*Note:* The present chapter has been typeset by giving unusual instructions to TgX so that it hyphenates words much more often than usual; therefore the reader can see numerous examples of word breaks that were discovered by the new algo rithm.  
*Chapter 2* 

The dictionary problem 

In this chapter we consider the problem of recognizing a set of words over a.u alphabet. To be more precise, an *alphabet* is a set of characters or symbols, for example the Liters A through Z, or the ASCII character set. A *word* is a sequence of characters from the alphabet. Given a set of words, our problem is to design a data structure that will allow us to determine efficiently whether or not some word is in the set. 

In particular, we will use spelling checking as an example throughout this chapter. This is a topic of interest in its own right, but we discuss it here because the pattern matching techniques we propose will turn out to be very useful in our hyphenation algorithm. 

Our problem is a special case of the general set recognition problem, because the elements of our set have the additional structure of being variable-length sequences of symbols from a finite alphabet. This naturally suggests methods based on a character-by-character examination of the key, rather than methods that operate on the entire key at once. Also, the redundancy present in natural languages such as English suggests additional opportunities for compression of the set representation. 

We will be especially interested in space minimization. Most data structures for set representation, including the one we propose, are reasonably fast for searching. That is, a search for a key doesn't take much more time than is needed to examine the key itself. However, most of these algorithms assume that everything is "in core", that is, in the primary memory of the computer. In many situations, such as our spelling checking example, this is not feasible. Since secondary memory access times are typically much longer, it is worthwhile to try compressing the data structure as much as possible. 

In addition to determining whether a given word is in the set, there arc other operations we might wish to perform on the set representation. The most basic are insertion and deletion of words from the set. More complicated operations include performing the union of two sets, partitioning a set according to some criterion,  
**:. THE DICTIONARY PROBLEM 9** 

determining which of several sets an element is a **member of, or operations based** on an ordering or other auxiliary information associated with the **keys** in **the set.** For the data structures we consider, we will pay some attention **to** methods **for** insertion and deletion, but we shall not discuss the **more complicated operations.** 

We first survey some known methods for set representation, and then **propose** a new data structure called a "packed trie". 

**Data structures** 

Methods for set representation include the following: **sequential lists, sorted** lists, binary search trees, balanced trees, hashing, superimposed coding, **bit** vec tors, and digital search trees (also known as tries). Good discussions of these data structures can be found in a number of texts, including Knuth \[4\], Standish \[5\], and AHU \[6\]. Below we make a few remsirks about each of these representations. 

A sequential list is the most straightforward representation. It requires both space and search time proportional to the number of characters in the dictionary. A sorted list assumes an ordering on the keys, such as alphabetical order. Binary search allows the search time to be reduced to the logarithm of the size of the dictionary, but space is not reduced. 

A binary search tree also allows search in logarithmic time. This can be thought of as a more flexible version of a sorted list that can be optimized in various ways. For example if the probabilities of searching for different keys in the tree are known, then the tree can be adapted to improve the expected search time. Search trees can also handle insertions and deletions easily, although an unfavorable sequence of such operations may degrade the performance of the tree. 

Balanced tree schemes (including AVL trees, 2-3 trees, and B-trees) correct the above-mentioned problem, so that insertions, deletions, and searches can all be performed in logarithmic time in the worst case. Variants of trees have other nice properties, too; they allow merging and splitting of sets, and priority queue operations. B-trees are well-suited to large applications, because they are designed to minimize the number of secondary memory accesses required to perform a search. However, space utilization is not improved by any of these tree schemes, and in fact it is usually increased because of the need for extra pointers. 

Hashing is an essentially different approach to the problem. Here a suitable randomizing function is used to compute the location at which a key is stored. Hashing methods arc very fast on the average, although the worst case is linear; fortunately this worst case almost never happens. 

An interesting variant of hashing, called superimposed coding, was proposed by Bloom \[7\] (see also \[4, §6.5\], \[8\]), and at last provides for reduction in space,  
**10 THE DICTIONARY PROBLEM** 

although at the expense of allowing some error. Since this method is perhaps less well known we give a description of it here. 

Superimposed coding 

The idea is as follows. We use a single large bit array, initialized to leros, plus a suitable set of *d* different hash functions. To represent a word, we use the hash functions to compute *d* bit positions in the large array of bits, and set these bits to ones. We do this for each word in the set. Note that some bits may be set by more than one word. 

To test if a word is in the set, we compute the *d* bit positions associated with the word as above, and check to see if they are all ones In the array. If any of them are zero, the word cannot be in the set, so we reject it. Otherwise if all of the bits are ones, we accept the word. However, some words not in the set might be erroneously accepted, if they happen to hash into bits that are all "covered" by words in the set. 

It can be shown \[7\] that the above scheme makes the best use of space when the density of bits in the array, after all the words have been inserted, is approximately one-half. In this case the probability that a word not in the set is erroneously accepted is *2\~d.* For example if each word is hashed into 4 bit positions, the error probability is 1/16. The required size of the bit array is approximately *ndlge,* where n is the number of items in the set, and lge « 1.44. 

In fact Bloom specifically discusses automatic hyphenation as an application for his scheme\! The scenario is as follows. Suppose we have a relatively compact routine for hyphenation that works correctly for.OO percent of the words in a large dictionary, but it is in error or fails to hyphenate the other 10 percent. We would then like some way to test if a word belongs to the 10 percent, but we do not have room to store all of these words in main memory. If we instead use the superimposed coding scheme to test for these words, the space required can be much reduced. For example with *d \=* 4 we only need aboxit 6 bits per wcrd. The penalty is that some words will be erroneously identified as being in the 10 percent. However, this is acceptable because usually the test word will be rejected and we can then be sure that it is not one of the exceptions. (Either it is in the other 90 percent or it is not in the dictionary at all.) In the comparatively rare case that the word is accepted, we can go to secondary store, to check explicitly if the word is one of the exceptions. 

The above technique is actually used in some commercial hyphenation routines. For now, however, T£JX will not have an external dictionary. Instead we will require that our hyphenation routine be essentially free of error (although it may not achieve complete hyphenation).  
**THE DICTIONARY PROBLEM 11** 

An extreme case of superimposed coding should also be mentioned, namely **the** bit-vector representation of a set. (Imagine that each word is associated with a single bit position, and one bit is allocated for each possible word.) This representation is often very convenient, because it allows set intersection and union to be performed by simple logical operations. But it also requires space proportional to **the** size of the universe of the set, which is impractical for words longer than three or four characters. 

**Tries** 

The final class of data structures we will consider are the digital **search trees,** first described by de la Briandais **\[9\]** and Frcdkin \[10\]. Fredkin also introduced the term "trie" for this class of trees. (The term was derived from the word retrieval, although it is now pronounced "try".) 

Tries are distinct from the other data structures discussed so far because they explicitly assume that the keys are a *sequence* of values over some (finite) alphabet, rather than a single indivisible entity. Thus tries are particularly well-suited for handling variable-length keys. Also, when appropriately implemented, tries can provide compression of the set represented, because common prefixes of words are combined together; words with the same prefix follow the same search path in the trie. 

A trie can be thought of as an m-ary tree, where m is the number of characters in the alphabet. A search is performed by examining the key one character at a time and using an m-way branch to follow the appropriate path in the trie, starting at the root. 

We will use the set of 31 most common English words, shown below, to illustrate different ways of implementing a trie. 

**`AAND ARE AS`** 

**`AT`**   
**`BE`**   
**`BUT`** 

**`FOR`**   
**`FROM HAD HAVE HE`**   
**`HER`**   
**`HIS`** 

**`IS IS`** 

**`IT`**   
**`NOT OF`** 

**`ON`**   
**`OR`** 

**`THE`**   
**`THIS TO`**   
**`WAS`** 

**`WHICH WITH YOU`** 

**BY I THAT** 

***Figure 4- The SI most common English words.***  
**12 THE DICTIONARY PROBLEM *Figure 5\. Linked trie for the SI most eommon English words.***  
**THE DICTIONARY PROBLEM 13** 

Figure 5 shows a *linked trie* representing this set of words. In a linked trie, the m-way branch is performed using a sequential scries of comparisons. Thus in Figure 5 each node represents a yes-no test against a particular character. There are two link fields indicating the next node to take depending on the outcome of the test. On a 'yes' answer, we also move to the next character of the key. The underlined characters are terminal nodes, indicated by an extra bit in the node. If the word ends when we are at a terminal node, then the word is in the set. 

Note that we do not have to actually store the keys in the trie, because each node^ implicitly represents a prefix of a word, namely the sequence of characters leading to that node. 

A linked trie is somewhat slow because of the sequential testing required for each character of the key. The number of comparisons per character can be as large as m, the size of the alphabet. In addition, the two link fields per node are somewhat wasteful of space. (Under certain circumstances, it is possible to eliminate ono of these two links. We will explain this later.) 

In an *indexed trie,* the m-way branch is performed using an array of size m. The elements of the array are pointers indicating the next family of the trie to go to when the given character is scanned, where a "family" corresponds to the group of nodes in a linked trie for testing a particular character of the key. When performing a search in an indexed trie, the appropriate pointer can be accessed by simply indexing from the base of the array. Thus search will be quite fast. 

But indexed tries typically waste a lot of space, because most of the arrays have only a few "valid" pointers (for words in the trie), with the rest of the links being null. This is especially common near the bottom of the trie. Figure 6 shows an indexed trie for the set of 31 common words. This representation requires 26 X 32 \= 

832 array locations, compared to 59 nodes for the linked trie. 

Various methods have been proposed to remedy the disadvantages of linked and indexed tries. Trabb Pardo \[11\] describes and analyzes the space requirements of some simple variants of binary tries. Knuth \[4, ex. 6.3-20\] analyzes a composite method where an indexed trie is used for the first few levels of the trie, switching to sequential search when only a few keys remain in a subtric. Mchlhorn \[12\] suggests using a I inary search tree to represent each family of a trie. This requires storage proportional to the number of "valid" links, as in a linked trie, but allows each character of the key to be processed in at most logm comparisons. Maly \[13\] has proposed a "compressed trie" that uses an implicit representation to eliminate links entirely. Each level of the trie is represented by a bit array, where the bits indicate whether or not some word in the set passes through the node corresponding to  
**14 THE DICTIONARY PROBLEM** 

`i` 

`2` 

`3` 

`4` 

`5` 

`6` 

`7` 

`8` 

`0` 

`10` 

`11` 

`12` 

`13` 

`14` 

`15` 

`16` 

`17` 

`18` 

`19` 

`20`   
`A`   
`2` 

`12`   
`B 5`   
`C`   
`D g` 

`g`   
`E` 

`0` 

`0` 

`14 g`   
`F`   
`7` 

`g`   
`G JI n` 

`21`   
`i` 

`16 15`   
`J K L M g`   
`N`   
`17 3` 

`g` 

`0`   
`0` 

`19` 

`8` 

`10` 

`18 g`   
`P Q R 4` 

`9` 

`g` 

**`o`** 

**`g`**   
**`s g`** 

**`o`** `g`   
`T`   
`20 g` 

`g` 

`g` 

`g`   
`U 6`   
`V` 

`13`   
`W`   
`24`   
`X Y 31` 

`g`   
**`z`** 

`21 22 23 24 25 26 27 28 29 30 31 32`   
`22 25` 

`28`   
`g` 

`26` 

`g` 

`g`   
`23` 

`29 27` 

`1` 

`32`   
`g g`   
`g` 

`30` 

`g` 

***Figure 6\. Indexed trie for the SI most eommon English words.***  
**THE DICTIONARY PRODLEM 15** 

that bit. In addition each family contains a field indicating the number of nonzero bits in the array for all nodes to the left of the current family, so that we can find the desired family on the next level. The storage required for each family is thus reduced to m+logn bits, where n is the total number of keys. However, compressed tries cannot handle insertions and deletions easily, nor do they retain the speed of indexed tries. 

Packed tries 

Our idea is to use an indexed trie, but to save the space for null links by packing the different families of the trie into a single large array, so that links from one family may occupy space normally reserved for links for other families that happen to be null. An example of this is illustrated below. 

**A \] G I C11 j E |** 

(In the following, we will sometimes refer to families of the indexed trie as *states,* and pointers as *transitions.* This is by analogy with the terminology for finite-state machines.) 

When performing a search in the trie, we need a way to check if an indexed pointer actually corresponds to the current family, or if it belongs to some other family that just happens to be packed in the same location. This is done by ad ditionally storing the character indexing a transition along with that transition. Thus a transition belongs to a state only if its character matches the character we are indexing on. This test always works if one additional requirement is satisfied, namely that different states may not be packed at the same base location. 

The trie can be packed using a first-fit method. That is, we pack the states one at a time, putting each state into the lowest-indexed location in which it will fit (not overlapping any previously packed transitions, nor at an already occupied base location). On numerous examples based on typical word lists, this heuristic works extremely well. In fact, nearly all of the holes in the trie are often filled by transitions from *c* ther states. 

Figure 7 shows the result when the indexed trie of Figure 6 is packed into a single array using the first-fit method. (Actually we have used an additional compression technique called suffix compression before packing the trie; this will be explained in the next section.) The resulting trie fits into just 60 locations. Note  
**16 THE DICTIONARY PROBLEM** 

**00**   
`0 1 A_8`   
`2` 

`Bll`   
`3 4 5 D_0`   
`6` 

`F 3`   
`7` 

`EjQ`   
`8` 

`H30`   
`9` 

`123` 

`C 5 H 0 N25 032 E 0 012 M 0` **10** 

**20 30**   
`T33 R 0`   
`R14 A29`   
`N 1 U 4`   
`W46 D 0`   
`T 0 S 0`   
`Y37 E12`   
`R 2 Y 0`   
**`s_o`** `N 0`   
`T 0 F_Q`   
`0 6 115` 

**40 50**   
`0 4 R 0`   
`H44 V 2`   
`S 0 038`   
`T 0 115`   
`I 7 H35`   
`A 4 136`   
`N 0 T 5`   
`A15 0 0 E 0 U_Q` 

***Figure 7\. Packed trie for the SI most common English word\*.*** 

that the packed trie is a single large array; the rows in the figure should be viewed as one long row. 

As an example, here's what happens when we search for tho word **HAVE** in **the** parked **trie. We** associate the values 1 through 26 with the letters A through **Z.** The **root of** the trie is packed at location 0, so we begin by looking at location 8 corresponding **to** the letter H. Since 'H30' is stored there, this is a valid transition and we then go to location 30\. Indexing by the letter A, we look in location 31, which tells us to go to 29\. Now indexing by V gets location 51, which points to 2\. Finally indexing by E gets location 7, which is underlined, indicating that the word HAVE is indeed in the set. 

Suffix **compression** 

**A** big advantage of the trie data structure is that common prefixes of words are combined automatically into common paths in the trie. This provides a good deal of compression. To save more space, we can try to take advantage of common suffixes.  
**THE DICTIONARY PROBLEM 17** 

One way of doing this is to construct a trie in the usual manner, and then merge common subtries together, starting from the leaves (lieves) and working upward. We call this process *suffix compression.* 

For example, in the linked trie of Figure 5 the terminal nodes **for the words** HIS and THIS, both of which test for the letter S and have no successors, can be combined into a single node. That is, we can let their parent nodes both point to the same node; this does not change the set of words accepted by the trie. It turns out that we can then combine the parent nodes, since both of them test for I and-go to the S node if successful, otherwise stop (no left successor). However, the grandparent nodes (which are actually siblings of the I nodes) cannot be combined even though they both test for E, because one of them goes to a terminal R node upon success, while the other has no right successor. 

With a larger set of words, a great deal of merging can be possible. Clearly all leaf nodes (nodes with no successors) that test the same character can be combined together. This alone saves a number of nodes equal to the number of words in the dictionary, minus the number of words that are prefixes of other words, plus at most 26\. In addition, as we might expect, longer suffixes such as \-ly, \-ing, or \-tion can frequently be combined. 

The suffix compression process may sound complicated, but actually it can be described by a simple recursive algorithm. For each node of the trie, we first compress each of its subtries, then determine if the node can be merged with some other node. In effect, we traverse the trie in depth-first order, checking each node to see if it is equivalent to any previously seen node. A hash table can be used to identify equivalent nodes, based on their (merged) transitions. 

The identification of nodes is somewhat easier using a binary tree representation of the trie, rather than an m-ary representation, because each node will then have just two link fields in addition to the character and output bit. Thus it will be convenient to use a linked trie when performing suffix compression. The linked representation is also more convenient for constructing the trie in the first place, because of the ease of performing insertions. 

After applying suffix compression, the trie can be converted to an indexed trie and packed as described previously. (We should remark that performing suffix compression on a linked trie can yield some addition?1 '.ompression, because trie families can be partially merged. However such compression is lost when the trie is converted to indexed form.) 

The author has performed numerous experiments with the above ideas. The re sults for some representative word lists are shown in Table 1 below. The last three  
**18 . THE DICTIONARY PROBLEM** 

columns show the number of nodes in the linked, suffix-compressed, and packed tries, respectively. Each transition of the packed trie consists of a pointer, a char acter, and a bit indicating if this is an accepting transition. 

word list words characters linked compressed packed 

pascal 35 145 125 murray 2720 19,144 8039 pocket 31,036 247,612 92,339 unabrd 235,545 2,250,805 759,045 

**`104`**   
**`4272`** 

**`38,619`**   
**`120`** 

**`4285`** 

**`38,638`** 

****Table 1\. Suffix-eompreised packed triei.* 

The algorithms for building a linked trie, suffix compression, and first-fit pack ing are used in Tj$X82 to preprocess the set of hyphenation patterns into a packed trie used by the hyphenation routine. A WEB description of these algorithms can be found in \[14\]. 

Derived forms 

Most dictionaries do not list the most common derived forms of words, namely regular plurals of nouns and verbs (-s forms), participles and gerunds of verbs (-ed and \-ing forms), and comparatives and superlatives of adjectives (-er and \-est). This makes sense, because a user of the dictionary can easily determine when a word possesses one of these regular forms. However, if we use the word list from a typical dictionary for spelling checking, we will be faced with the problem of determining when a word is one of these derived forms. 

Some spelling checkers deal with this problem by attempting to recognize af fixes. This is done not only for the derived forms mentioned above but other com mon variant forms as well, with the purpose of reducing the number of word3 that have to be stored in the dictionary. A set of logical rules is used to determine when certain prefixes and suffixes can be stripped from the word under consideration. 

However such rules can be quite complicated, and they inevitably make errors. The situation is not unlike that of finding rules for hyphenation, which should not be surprising, since affix recognition is an important part of any rule-based hyphenation algorithm. This problem has been studied iu some detail in a series of papers by Resnikoff and Dolby \[15\]. 

Since affix recognition is difficult, it is preferable to base a spelling checker on a complete word list, including all derived forms. However, a lot of additional space will be required to store all of these forms, even though much of the added data is  
THE DICTIONARY PROBLEM 19 

redundant. We might hope that some appropriate method could provide substan tial compression of the expanded word list. It turns out that suffix-compressed tries handle this quite well. When derived forms were added to our pocket dictionary word list, it increased in size to 49,858 words and 404,046 characters, but the result ing packed trie only increased to 46,553 transitions (compare the pocket dictionary statistics in Table 1). 

"Hyphenation programs also need to deal with the problem of derived forms. In our pattern-matching approach, we intend to extract the hyphenation rules au tomatically from the dictionary. Thus it is again preferable for our word list to include all derived forms. 

The creation of such an expanded word list required a good deal of work. The author had access to a computer-readable copy of Webster's Pocket Dictionary \[16\], including parts of speech and definitions. This made it feasible to identify nouns, verbs, etc., and to generate the appropriate derived forms mechanically. Unfortunately the resulting word lists required extensive editing to eliminate muny never-used or somewhat nonsensical derived forms, e.g. 'informations'. 

Spelling checkers 

Computer-based word processing systems Lave recently come into widespread use. As a result there has been a surge of interest in programs for automatic spelling checking and correction. Here we will consider the dictionary representations used by some existing spelling checkers. 

One of the earliest programs, designed for a large timesharing computer, was the DEC-10 SPELL program written by Ralph Gorin \[17). It uses a 12,000 word dictionary stored in main memory. A simple hash function assigns a unique 'bucket\* to each word depending on its length and the first two characters. Words in the same bucket are listed sequentially. The number of words in each bucket is relatively small (typically 5 to 50 words), so this representation is fairly efficient for searching. In addition, the buckets provide convenient access to groups of similar words; this is useful when the program tries to correct spelling errors. 

The dictionary used by SPELL does not contain derived forms. Instead some simple affix stripping rules arc normally used; the author of the program notes that these are "error-prone". 

Another spelling checker is described by James L. Peterson \[18\]. His program uses three separate dictionaries: (1) a small list of 258 common English words, (2) a dynamic 'cache' of about 1000 document-specific words, and (3) a large, compre hensive dictionary, stored on disk. The list of common words (which is static) is represented using a suffix-compressed linked trie. The dynamic cache is maintained  
**20 THE DICTIONARY PROBLEM** 

using a hash table. Both of these dictionaries arc kept in main memory for speed. The disk dictionary uses an in-core index, so that at most one disk access is required per search. 

Robert Nix \[19\] describes a spelling checker based on the superimposed coding method. He reports that this method allows the dictionary from the SPELL pro gram to be compressed to just 20 percent of its original size, while allowing 0.1% chance of error. 

A considerably different approach to spelling checking was taken by the TYPO program developed at Bell Labs \[20\]. This program uses digram and trigram fre quencies to identify "improbable" words. After processing a document, the words are listed in order of decreasing improbability for the user to peruse. (Words ap pearing in a list of 2726 common technical words are not shown.) The authors report that this format is "psychologically rewarding", because many errors are found at the beginning, inducing the user to continue scanning the list until errors become rare. 

In addition to the above, there have recently been a number of spelling checkers developed for the "personal computer" market. Because these programs run on small microprocessor-based systems, it is especially important to reduce the size of the dictionary. Standard techniques include hash coding (allowing some error), in core caches of common words, and special codes for common prefixes and suffixes. One program first constructs a sorted list of all words in the document, and then compares this list with the dictionary in a single sequential pass. The dictionary can then be stored in a compact form suited for sequential scanning, where each word is represented by its difference from the previous word. 

Besides simply detecting when words are not in a dictionary, the design of a practical spelling checker involves a number of other issues. For example many spelling checkers also try to perform spelling *correction.* This is usually done by searching the dictionary for words similar to the misspelled word. Errors and sug gested replacements can be presented in an interactive fashion, allowing the user to see the context from the document and make the necessary changes. The contents of the dictionary arc of course very important, and each user may want to modify the word list to match his or her own vocabulary. Finally, a plain spelling checker cannot detect problems such as incorrect word usage or mistakes in grammar; a more sophisticated program performing syntactic and perhaps semantic analysis of the text would be necessary.  
**THE DICTIONARY PROBLEM 21** 

Conclusion and related ideas 

The dictionary problem is a fundamental problem of computer science, and it has many applications besides spelling checking. Most data structures for this problem consider the elements of the set as atomic entities, fitting into a single com puter word. However in many applications, particularly word processing, the keys are actually variable-length strings of characters. Most of the standard techniques are somewhat awkward when dealing with variable length keys. Only the trie data structure is well-suited for this situation. 

We have proposed a variant of tries that we call a packed trie. Search in a packed trie is performed by indexing, and it is therefore very fast. The first-fit packing technique usually produces a fairly compact representation as well. 

We have not discussed how to perform dynamic insertions and deletions with a packed trie. In Chapter 4 we discuss a way to handle this problem, when no suffix compression is used, by repacking states when necessary. 

The idea of suffix compression is not new. As mentioned, Peterson's spelling checker uses this idea also. But in fact, if we view our trie as a finite-state machine, suffix compression is equivalent to the well-known idea of state minimization. In our case the machine is acyclic, that is, it has no loops. 

Suffix compression is also closely related to the common subexpression problem from compiler theory. In particular, it can be considered a special case of a problem called acyclic congruence closure, which has been studied by Downey, Sethi, and Tarjan \[21\]. They give a linear-time algorithm for suffix compression that does not use hashing, but it is somewhat complicated to implement and requires additional data structures. 

The idea for the first-fit packing method was inspired by the paper "Storing a sparse table" by Tarjan and Yao \[22\]. The technique has been used for compressing parsing tables, as discussed by Zeiglcr \[23\] (see also \[24\]). However, our packed trie implementation differs somewhat from the applications discussed in the above references, because of our emphasis on space minimization. In particular, the idea of storing the character that indexes a transition, along with that transition, seems to be new. This has an advantage over other techniques for distinguishing states, such as the use of back pointers, because the character requires fewer bits. 

The paper by Tarjan and Yao also contains on interesting theorem character izing the performance of the first-fit packing method. They consider a modification suggested by Zcigler, where the states arc first sorted into decreasing order based on the number of non-null transitions in each state. The idea is that small states, which can be packed more easily, will be saved to the end. They prove that if the  
**22 THIS DICTIONARY PROBLEM** 

distribution of transitions among states satisfies a "harmonic decay" condition, then essentially all of the holes in the first-fit packing will be filled. 

More precisely, let n(/) be the total number of non-null transitions in states with more than / transitions, for / \> 0\. If the harmonic decay property *n(l) \< n/(l* \-f 1\) is satisfied, then the first-fit-decrcasing packing satisfies 0 \< *b(i) \<* n for all •', where *n* \= n(0) is the total number of transitions and *b(i)* is the base location at which the tth state is packed. 

The above theorem does not take into account our additional restriction that no two states may be packed at the same base location. When the proof is modified to include this restriction, the bound goes up by a factor of two. However in practice we seem to be able to do much better. 

The main reason for the good performance of the first-fit packing scheme is the fact that there are usually enough single-transition states to fill in the holes created by larger states. It is not really necessary to sort the states by number of transitions; any packing order that distributes large and small states fairly evenly will work well. We have found it convenient simply to use the order obtained by traversing the linked trie. 

Improvements on the algorithms discussed in this chapter are possible in certain cases. If we store a linked trie in a specific traversal order, we can eliminate one of the link fields. For example, if we list the nodes of the trie in preordcr, the left successor of a node will always appear immediately after that node. An extra bit is used to indicate that a node has no left successor. Of course this technique works for other types of trees as well. . . 

If the word list is already sorted, linked trie insertion can be performed with only a small portion of the trie in memory at any time, namely the portion along the current insertion path. This can be a great advantage if we are are processing a large dictionary and cannot store the entire linked trie in memory.  
***Chapter 3*** 

Hyphenation 

**Let us now try to apply the ideas of the previous chapter to the problem of** hyphenation. T£JX82 will use the pattern matching **method described in Chapter 1,** but we shall first discuss some related approaches that **were considered. Finite-state machines with output**   
We can modify our trie-based dictionary representation **to perform hyphenation** by changing the output of the trie (or finite-state machine) **to a** multiple-valued output indicating how the word can be hyphenated, instead **of** just **a** binary **yes-no** output indicating whether or not the word is in the dictionary. That **is,** instead of associating a single bit with each trie transition, we would have a larger "output" field indicating the hyphenation "action" to be taken on this transition. Thus on recognizing the word hy-phen-a-tion, the output would say "you can hyphenate this word after the second, sixth, or seventh letters". 

To represent the hyphenation output, we could simply list **the** hyphen positions, or we could use a bit vector indicating the allowable hyphen points. Since there arc only a few hundred different outputs and most of them occur many times, **we** can save some space by assigning each output a unique code and storing the actual hyphen positions in a separate table. 

To conveniently handle the variable number of hyphen positions in outputs, we will use a linked representation that allows different outputs to share common portions of their output lists. This is implemented using a hash table containing pairs of the form *(output, next),* where *output* is a hyphenation position and *next* is a (possibly null) pointer to another entry in the table. To add a new output list to the table, we hash each of its outputs in turn, making each output point to the previous one. Interestingly, this process is quite similar to suffix compression. 

The trie with hyphenation output can be suffix-compressed and packed in the same manner as discussed in Chapter 2\. Because of the greater variety of out puts more of the subtrics will be distinct, and there is somewhat less compression. 

**23**  
24 HYPHENATION 

From our pocket dictionary (with hyphens), for example, we obtained a packed trie occupying 51,699 locations. 

We can improve things slightly by "pushing outputs forward". That is, we can output partial hyphenations as soon as possible instead of waiting until the end of the word. This allows some additional suffix compression. 

For example, upon scanning the letters hyph at the beginning of a word, we can already say "hyphenate after the second letter" because this is allowed for all words beginning with those letters. Note we could not say this after scanning j . *jt* hyp, because of words like hyp-not-ic. Upon further scanning ena, we can say "hyphenate after the sixth letter". 

When implementing this idea, we run into a small problem. There are quite a few words that are prefixes of other words, but hyphenate differently on the letters they have in common, e.g. ca-ret and care-tak-er, or aa-pi-rin and aa pir-ing. To avoid losing hyphenation output, we could have a separate output whenever an end-of-word bit appears, but a simpler method is to append an end-of word character to each word before inserting it into the trie. This increases the size of the linked trie considerably, but suffix compression merges most of these nodes together. 

With the above modifications, the packed trie for the pocket dictionary was reduced to 44,128 transitions. 

Although we have obtained substantial compression of the dictionary, the result is still too large for our purposes. The problem is that as long as we insist that only words in the dictionary be hyphenated, we cannot hope to reduce the space required to below that needed for spelling checking alone. So we must give up this restriction. 

For example, we could eliminate the end-of-word bit. Then after pushing out puts forward, we can prune branches of the trie for which there is no further output. This would reduce the pocket dictionary trie to 35,429 transitions. 

Minimization with don't cares / 

In this section we describe a more drastic approach to compression that takes advantage of situations where we "don't care" what the algorithm dors. As previously noted, most of the states in an indexed trie are quite sparse; that is, only a few of the characters have explicit transitions. Since the missing transitions are never accessed by words in our dictionary, we can allow them to be filled by arbitrary transitions.  
**HYPHENATION 25** 

This should not be confused with the overlapping of states that may occur in the trie-packing process. Instead, we mean that the added transitions will actually become part of the state. 

There are two ways in which this might allow us to save more space in the min imization process. First, states no longer have to be identical in order to be merged; they only have to agree on those characters where both (or all) have explicit transi tions. Second, the merging of non-equivalent states may allow further merging that was not previously possible, because some transitions have now become equivalent. 

For example, consider again the trie of Figure 5\. When discussing suffix com pression, we noted that the terminal S nodes for the words HIS and THIS could be merged together, but that the parent chains, each containing transitions for A, E, and I, could not be completely merged. However, in minimization with don't cares these two states can be merged. Note that such a merge will require that the DV state below the first A be merged with the T below the second A; this can be done 

because those states have no overlapping transitions. 

As another example, notice that if the word AN were added to our vocabulary, then the NRST chain succeeding the root A node could be merged with the NST chain below the initial I node. (Actually, it doesn't make much sense to do minimization with don't cares on a trie used to recognize words in a dictionary, but we will ignore that objection for the purposes of this example.) 

Unfortunately, trie minimization with don't cares seems more complicated than the suffix-compression process of Chapter 2\. The problem is that states can be merged in more than one way. That is, the collection of mcrgcable states no longer forms an equivalence relation, as in regular finite-state minimization. In fact, we can sometimes obtain additional compression by allowing the same state to appear more than once. Another complication is that don't care merges can introduce loops into our trie. 

Thus it seems that finding the minimum size trie will be difficult. Pfleeger \[25\] has shown this problem to be NP-complete, by transformation from graph coloring; however, his construction requires the number of transitions per state to be unbounded. It may be possible to remove this requirement, but we have not proved this. 

So in order to experiment with trie minimization with don't cares, we have made some simplifications. We start by performing suffix compression in the usual manner. We then go through the states in a bottom-up order, checking each to see if it can be merged with any previous state by taking advantage of don't cares. Note that such merges may require further merges among states already seen.  
**28 HYPHENATION** 

We only try merges that actually save space, that is, **where explicit transitions are** merged. Otherwise, states with only a few transitions **are very likely to be** mergeable, but such merges may constrain us unnecessarily **at a later stage of the** minimization. In addition, we will not consider having **multiple copies of states.** 

Even this simplified algorithm can be quite time consuming, so we did **not try** it on our pocket dictionary. On a list of 2726 technical **words, don't care minimization reduced** the number **of** states in the suffix-compressed, **output-pruned trie from 1685 to** just 283, while the number of transitions was **reduced from 3627 to 2427\.** However, because the resulting states were larger, **the first-fit** packing **performed** rather poorly, producing a packed trie with 3408 transitions. **So in this case don't care** minimization yielded an additional compression of less than 10 percent. 

Also, the behavior of the resulting hyphenation algorithm **on words not in the** dictionary became rather unpredictable. Once **a word leaves the "known" paths of the** packed "trie, strange things might happen\! 

We can get even wilder effects by carrying **the don't care assumption one step further,** and eliminating the character field **from the packed trie altogether (leaving** just the output and trie link). Words in the dictionary will always index **the correct** transitions, but on other words we now have no way of telling when we have reached an invalid trie transition. 

It turns out that the problem of state minimization with don't cares was studied in the 1960s by electrical engineers, who called it "minimization of incompletely specified sequential machines" (see e.g. \[26\]). However, typical instances of the problem involved machines with only a few states, rather than thousands as in our case, so it was often possible to find a minimized machine by hand. Also, the emphasis was on minimizing the number of states of the machine, rather than the number of state transitions. 

In ordinary finite-state minimization, these are equivalent, but don't care min imization can actually introduce extra transitions, for example when states are duplicated. In the old days, finite-state machines were implemented using combina tional logic, so the most important consideration WJXS to reduce the number of states. 

In our trie representation, however, the space used is proportional to the number of transitions. Furthermore, finite-state machines are now often implemented using PLA's (programmed logic arrays), for which the number of transitions is also the best measure of space. 

Pattern matching 

Since trie minimization with don't cares still doesn't provide sufficient compres sion, and since it lead • to unpredictable behavior on words not in the dictionary,  
**HYPHENATION 27** 

we need a different approach. It seems expensive to insist on complete hyphenation of the dictionary, *so* we will give up this requirement. We could allow some errors; or to be safer, we could allow some hyphens to be missed. 

We now return to the pattern matching approach described in Chapter 1\. Some further arguments as to why this method seems advantageous are given below. We should first reassure the reader that all the discussion so far has not been in vain, because a packed trie will be an ideal data structure for representing the patterns in the final hyphenation algorithm. Here the outputs will include the hyphenation level as well as the intercharacter position. 

Hyphenating and inhibiting patterns allow considerable flexibility in the per formance of the resulting algorithm. For example, we could allow a certain amount of error by using patterns that aren't always safe (but that presumably do find many correct hyphens). 

We can also restrict ourselves to partial hyphenation in a natural way. That is, it turns out that a relatively small number of patterns will get a large fraction of the hyphens in the dictionary. The remaining hyphens become harder and harder to find, as we are left with mostly exceptional cases. Thus we can choose the most effective patterns first, taking more and more specialized patterns until we run out of space. 

In addition, patterns perform quite well on words not in the dictionary, if those words follow "normal" pronunciation rules. 

Patterns are "context-free"; that is, they can apply anywhere in a word. This seems to be an important advantage. In the trie-based approach discussed earlier in this chapter, a word is always scanned from beginning to end and each state of the trie 'remembers' the entire prefix of the word scanned so far, even if the letters scanned near the beginning no longer affect the hyphenation of the word. Suffix compression eliminates some of this unnecessary state information, by combining states that are identical with respect to future hyphenation. Minimization with don't cares takes this further, allowing 'similar' states to be combined as long as they behave identically on all characters that they have in common. 

However, we have seen that it is difficult to guide the minimization with don't cares to achieve these reductions. Patterns embody such don't care situations nat urally (if we can find a good way of selecting the patterns). 

The context-free nature of patterns helps in another way, as explained below. Recall that we will use a packed trie to represent the patterns. To find all patterns that match in a given word, we perform a search starting at each letter of the word. Thus after completing a search starting from some letter position, we may have to  
**28 HYPHENATION** 

back up in the word to start the next search. By contrast, our original trio-based approach works with no backup. 

Suppose we wanted to convert the pattern trie into a finite-state recognizer that works with no backup. This can be done in two stages. We first add "failure links" to each state that tell which state to go to if there is no explicit transition for the current character of the word. The failure state is the state in the trie that we would have reached, if we had started the search one letter later in the word. 

Next, we can convert the failure-link machine into a true finite-state machine by-filling in the missing transitions of each state with those of its failure state. (For more details of this process, see \[27\], \[28\].) 

However, the above state merging will introduce a lot of additional transitions. Even using failure links requires one additional pointer per state. Thus by perform ing pattern matching with backup, we seem to save a good deal of space. And in practice,'long backups rarely occur. 

Finally, the idea of inhibiting patterns seems to be very useful. Such patterns extend the power of a finite-state machine, somewhat like adding the "not" operator to regular expressions.  
***Chapter 4*** 

Pattern generation 

We now discuss how to cboose a suitable set of patterns for hyphenation. In or der to decide which patterns are "good", we must first specify the desired properties of the resulting hyphenation Jgorithm. 

We obviously want to maximize the number of hyphens found, minimize the error, and minimize the space required by our algorithm. For example, we could try to maximize some (say linear) function of the above three quantities, or we could hold one or two of the quantities constant and optimize the others. 

For 1^X82, we wanted a hyphenation algorithm meeting the following require ments. The algorithm should use only a moderate amount of space (20-30K bytes), including any exception dictionary; and it should find as many hyphens as possible, while making little or no error. This is similar to the specifications for the original TjjX algorithm, except that we now hope to find substantially more hyphens. 

Of course, the results will depend on the word list used. We decided to base the algorithm on our copy of Webster's Pocket Dictionary, mainly because this was the only word list we had that included all derived forms. 

We also thought that a larger dictionary would contain many rare or specialized words that we might not want to worry about. In p' ticular, we did not want such infrequent words to affect the choice of patterns, because we hoped to obtain a set of patterns embodying many of the "usual" rules for hyphenation. 

In developing the Tj\<jX82 algorithm, however, the word list was tuned up con siderably. A few thousand common words were weighted more heavily so that they would be more likely to be hyphenated. In fact, the current algorithm guarantees complete hyphenation of the 676 most common English words (according to \[29\]), as well as a short list of common technical words (e.g. al-go-rithm). 

In addition, over 1000 "exception" words have been added to the dictionary, to ensure that they would not be incorrectly hyphenated. Most of these were found by testing the algorithm (based on the initial word list) against a larger dictionary obtained from a publisher, containing about 115,000 entries. This produced about 

29  
**30 PATTERN GENERATION** 

10,000 errors on words not in the pocket dictionary. Most of these were specialised technical terms that we decided not to worry about, but a few hundred were em barrassing enough that we decided to add them to the word list. These included compound words (camp-fire), proper names (Af-ghan-i-stan), and new words (bio-rhythm) that probably did not exist in 1966, when our pocket dictionary was originally put online. 

After the word list was augmented, a new set of patterns was generated, and a new list of exceptions was found and added to the list. Fortunately this process seemed to converge after a few iterations. 

Heuristics 

The selection of patterns in an 'optimal' way seems very difficult. The problem is that f cvera\! patterns may apply to a particular hyphen point, including both hyphenating and inhibiting patterns. Thus complicated interactions can arise if we try to determine, say, the minimum set of patterns finding a given number of hyphens. (The situation is somewhat analogous to a set cover problem.) 

Instead, we will select patterns in a series of "passes" through the word list. In each pass we take into account only the effects of patterns chosen in previous passes. Thus we sidestep the problem of interactions mentioned above. 

In addition, we will define a measure of pattern "efficiency" so that we can use a greedy approach in each pass, selecting the most efficient patterns. Patterns will be selected one level at a time, starting with a level of hyphenating patterns. Patterns at each level will be selected in order of increasing pattern length. Furthermore patterns of a given length applying to different intercharacter positions (for example \-ti o and t-io) will be selected in separate passes through the dictionary. Thus the patterns of length n at a given level will be chosen in n \-f 1 passes through the dictionary. 

At first we did not do this, but selected all patterns of a given length (at a given level) in a single pass, to save time. However, we founJ that this resulted in considerable duplication of effort, as many hyphens were covered by two or more patterns. By considering different intercharacter positions in separate passes, there is never any overlap among the patterns selected in a single pass. 

In each pass, we collect statistics on all patterns appearing in the dictionary, counting the number of times we could hyphenate at a particular point in the pattern, and the number of times we could not. 

For example, the pattern tio appears 1793 times in the pocket dictionary, and in 1773 cases we can hyphenate the word before the t, while in 20 cases we can  
**PATTERN GENERATION 31** 

**not.** (We only count instances where the hyphen position **occurs at least two letters** from either edge of the word.) 

These counts **arc** used to determine the efficiency **rating of patterns. For exam\* pie if we arc** considering only "safe" patterns, that is, **patterns that can always be hyphenated at a** particular position, then a reasonable **rating is simply the number of hyphens found.** We could then decide **to** take, **say, all patterns finding at least a** given **number of** hyphens. 

However, most of the patterns we use will make some **error. How should these** patterns be evaluated? In the worst case, errors can **be handled by simply** listing them in an exception dictionary. Assuming that one unit of space is required to represent each pattern as well as each exception, the "efficiency" of **a** pattern could be defined as *eff= good* /(I \-f *bad)* where *good* is the number of hyphens correctly 

found and *bad* is the number of errors made. 

(The space used by the final algorithm really depends on how much compression is produced by the packed trie used to represent the patterns, but **since** it is hard to predict the exact number of transitions required, we just use the number of patterns as an approximate measure of size.) 

By using inhibiting patterns, however, we can often do better than listing the exceptions individually. The quantity *bad* in the above formula should then be devalued a bit depending on how effective patterns at the next level are. So **a** better formula might be 

e ir \= *W\**   
•" *1 \+ bad/bad-eff'* 

where *bad.ejj* is the estimated efficiency of patterns at **the next level** (inhibiting errors at the current level). 

Note that it may be difficult to determine the efficiency at the next level, when we are still deciding what patterns to take at the current level\! We will use **a** pattern selection criterion of the form *eff\> thresh,* but we cannot predict exactly how many patterns will be chosen and what their overall performance will be. The best we can do is use reasonable estimates based on previous runs of the pattern generation program. Some statistics from trial runs of this program are presented later in this chapter. 

Collecting pattern statistics 

So the main task of the pattern generation process is to collect count statistics about patterns in the dictionary. Because of time and space limitations this becomes an interesting data structure exercise.  
**32 PATTERN GENERATION**   
• 

For short (length 2 and 3\) patterns, we can simply use a table of size 26a or 263, respectively, to hold the counts during a pass through the dictionary. For longer patterns, this is impractical. 

Here's the first approach we used for longer patterns. In a pass through the dictionary, every occurrence of a pattern is written out to a file, along with an indi cation of whether or not a hyphen was allowed at the position under consideration. The file of patterns is sorted to bring identical patterns together, and then a pass is made through the sorted list to compile the count statistics for each pattern. 

This approach makes it feasible to collect statistics for longer length patterns, and was used to conduct our initial experiments with pattern generation. However it is still quite time and space consuming, especially when sorting the large lists of patterns. Note that an external sorting algorithm is usually necessary. 

Since only a fraction of the possible patterns of a particular length actually occur in the dictionary, we could instead store them in a hash tabls or one of the other data structures discussed in Chapter 2\. It turns out that a modification of our packed trie data structure is well-suited to this task. The advantages of the packed trie are very fast lookup, compactness, and graceful handling of variable length patterns. 

Combined with some judicious "pruning" of the patterns that are considered, the memory requirements are much reduced, allowing the entire pattern selection process to be carried out "in core" on our PDP-10 computer. 

By "pruning" patterns we mean the following. If a pattern contains a shorter pattern at the same level that has already been chosen, the longer pattern obviously need not be considered, so we do not have to count its occurrences. Similarly, if a pattern appears so few times in the dictionary thtt under the current selection criterion it can never be chosen, then we can mark the pattern as "hopeless" so that any longer patterns at this level containing it need not be considered. 

Pruning greatly reduces the number of patterns that must be considered, es pecially at longer lengths. 

Dynamic packed tries 

Unlike the static dictionary problem considered in Chapter 2, the set of patterns to be represented is not known in advance. In order to use a packed trie for storing the patterns being considered in a pass through the dictionary, we need some way to dynamically insert new patterns into the trie. 

For any pattern, we start by performing a search in the packed trie as usual, following existing links until reaching a state where a new trie transition must be  
**PATTERN GENERATION 33** 

added. If we are lucky, the location needed by the new transition will still be empty in the packed trie, otherwise we will have to do some repacking. Note that we will not be using suffix compression, because this complicates things considerably. We would need back pointers or reference counts to determine what nodes need to be unmerged, and we would need a hash table or other auxiliary information in order to remerge the newly added nodes. Furthermore, suffix merging does not produce a great deal of compression on the relatively short patterns we will be dealing with. 

The simplest way of resolving the packing conflict caused by the addition of a new transition is to just repack the changed state (and update the link of its parent state). To maintain good space utilization, we should try to fit the modified Btate among the holes in the trie. This can be done by maintaining a dynamic list of unoccupied cells in the trie, and using a first-fit search. 

However, repacking turns out to be rather expensive for large states that are unlikely to fit into the holes in the trie, unless the array is very sparse. We can avoid this by packing such states into the free space immediately to the right of the occupied locations. The size threshold for attempting a first-fit packing can be adjusted depending on the density of the array, how much time we are willing to spend on insertions, or how close we are to running out of room. 

After adding the critical transition as discussed above, we may need to add some more trie nodes for the remaining characters of the new pattern. These new states contain just a single transition, so they should be easy to fit into the trie. 

The pattern generation program uses a second packed trie to store the set of patterns selected so far. Recall that, before collecting statistics about the patterns in each word, we must first hyphenate the word according to the patterns chosen in previous passes. This is done not only to determine the current partial hyphenation, but also to identify pruned patterns that need not be considered. Once again, the advantages of the packed trie are compactness and very fast "hyphenation". 

At the end of a pass, we need to add new patterns, including "hopeless" pat terns, to the trie. Thus it will be convenient to use a dynamic packed trie here as well. At the end of a level, we probably want to delete hopeless patterns from the trie in order to recover their space, if we are going to generate more levels. This turns out to be relatively easy; we just remove the appropriate output and return any freed nodes to the available list. 

Below we give some statistics that will give an idea of how well a dynamic packed trie performs. We took the current set of 4447 hyphenation patterns, ran domized them, and then inserted them one-by-one into a dynamic packed trie.  
**34 PATTERN GENERATION** 

(Note that in the situations described above, there will actually be many searches per insertion, so we can afford some extra effort when performing insertions.) The patterns occupy 7214 trie nodes, but the packed trie will use more locations, de pending on the setting of the first-fit packing threshold. The columns of the table show, respectively, the maximum state size for which a first-fit packing is attempted, the number of states packed, the number of locations tried by the first-fit procedure (this dominates the running time), the number of states repacked, and the number of locations used in the final packed trie. 

thresh pack first-fit unpack trie\_max 

oo 6113 877,301 2781 9671 

13 6060 761,228 2728 9458 

9 6074 559,835 2742 9606 

7 6027 359,537 2695 9006 

5 5863 147,468 2531 10,366 

4 5746 03,181 2414 11,209 

3 5563 33.82G 2231 13,296 

2 5242 10,885 1910 15,009 

1 4847 895C 1515 16,536 

0 4577 6073 1245 18,628 

*Table 2\. Dynamic packed trie statistic\*.* 

Experimental results 

We now give some results from trial runs of the pattern generation program,' and explain how the current 1^X82 patterns were generated. As mentioned earlier, the development of these patterns involved some augmentation of the word list. The results described here arc based on the latest version of the dictionary. 

At each level, the selection of patterns is controlled by three parameters called *good-wt, bad.wt,* and *thresh.* If a pattern can be hyphenated *good* times at a partic ular position, but makes *bad* errors, then it will be selected if 

*good\* good.wt — bad\* bad.wt \> thresh.* 

Note that the efficiency formula given earlier in this chapter can be converted into the above form. 

We can first try using only safe patterns, that is, patterns that can always be hyphenated at a particular position. The table below shows the results when all safe patterns finding at least a given number of hyphens are chosen. Note that  
*gf* PATTERN GENERATION 35 parameters patterns hyphens percent   
1 1 1 1 1 1 1   
oo **CO CO** oo oo oo **CO**   
40 20 10 5 

3 

2 

1   
401 

1024 

2272 

4603 

7052 

10,456 16,336 

31,083 45,310. 58,580 70,014 76,236 83,450 87,271 

35.2% 51.3% 66.3% 79.2% 86.2% 94.4% 98.7% 

*Table S. Safe hyphenating patterns.* 

an infinite *bad.wt* ensures that only safe patterns are chosen. The table shows the number of patterns obtained, and the number and percentage of hyphens found. We see that, roughly speaking, halving the threshold doubles the number of patterns, but only increases the percentage of hyphens by a constant amount. The last 20 percent or so of hyphens become quite expensive to find. (In order to save computer time, we have only considered patterns of length 6 or less in obtaining the above statistics, so the figures do not quite represent all patterns above a given threshold. In particular, the patterns at threshold 1 do not find 100% of the hyphens, although even with indefinitely long patterns there would still be a few hyphens that would not be found, such as re-cord.) The space required to represent patterns in the final algorithm is slightly more than one trie transition per pattern. Each transition occupies 4 bytes (1 byte each for character and output, plus 2 bytes for trie link). The output table requires an additional 3 bytes per entry (hyphenation position, value, and next output), but there are only a few hundred outputs. Thus to stay within the desired space limitations for TjrjX82, we can use at most about 5000 patterns. We next try using two levels of patterns, to see if the idea of inhibiting patterns actually pays off. The results are shown below, where in each case the initial level of hyphenating patterns is followed by a level of inhibiting patterns that remove nearly all of the error. 

The last set of patterns achieves 86.7% hyphenation using 4696 patterns. By contrast, the 1 oo 3 patterns from the previous table achieves 86.2% with 7052 patterns. So inhibiting patterns do help. In addition, notice that we have only used "8\<afc" inhibiting patterns above; this means that none of the good hyphens are lost. We can do better by using patterns that also inhibit some correct hyphens. 

After a good deal of further experimentation, we decided to use *Rve* levels of patterns in the current T\]rjX82 algorithm. The reason for this is as follows. In  
**36 PATTERN GENERATION** 

parameters patterns hyphens percent 

51,359 505 58.1% 0.6% 

0 463 58.1% 0.1% 

64,893 1694 73.5% 1.9% 

0 1531 73.5% 0.2% 

76,632 5254 86.7% 5.9% 

0 4826 86.7% 0.5% 

***Table 4- Two levels of patterns.*** 

addition to finding a high percentage of hyphens, we also wanted a certain amount of guaranteed behavior. That is, we wanted to make essentially no errors on words in the dictionary, and also to ensure complete hyphenation of certain common words. 

To accomplish this, we use a final level of safe hyphenating patterns, with the threshold set as low as feasible (in our case 4). If we then weight the list of important words by a factor of at least 4, the patterns obtained will hyphenate them completely (except when a word can be hyphenated in two different ways). 

To guarantee no error, the level of inhibiting patterns immediately preceding the final level should have a threshold of 1 so that even patterns applying to a single word will be chosen. Note these do not need to be "safe" inhibiting patterns, since the final level will pick up all hyphens that should be found. 

The problem is, if there are too many errors remaining before the last inhibiting level, we will need too many patterns to handle them. If we use three levels in all, then the initial level of hyphenating patterns can allow just a small amount of error. 

However, we would like to take advantage of the high efficiency of hyphenating patterns that allow a greater percentage of error. So instead, we will use an initial level of hyphenating patterns with relatively high threshold and allowing consider able error, followed by a 'coarse' level of inhibiting patterns removing most of the initial error. The third level will consist of relatively safe hyphenating patterns with a somewhat lower threshold than the first level, and the last two levels will be as described above. 

The above somewhat vague considerations do not specify the exact pattern selection parameters that should be used for each pass, especially the first three passes. These were only chosen after much trial and error, which would take too long to describe here. We do not have any theoretical justification for these parameters; they just seem to work well. 

The table below shows the parameters used to generate the current set of TgX82 patterns, and the results obtained. For levels 2 and 4, the numbers in the "hyphens"  
PATTERN GENERATION 87 

level p^arameters patterns hyphens percent 1 1 2 20 (4) 458 67,604 14,156 76.6% 16.0%   
2 3 4 5   
2 

1 

3 

1 

1 

4 

2 

oo 

8 

7 

1 

4   
(4) 

(5) 

(6) 

(8) 

*Table* 

509 

985 

1647 

1320 

*S. Current*   
7407 11,942 68.2% 2.5% 13,198 551 83.2% 3.1% 1010 2730 82.0% 0.0% 6428 0 89.3% 0.0% 

*TEX82 pattern\*.* 

column show the number of good and bad hyphens inhibited, respectively. **The** numbers in parentheses indicate the maximum length of patterns chosen at that level. 

A total of 4919 patterns (actually only 4447 because some patterns appear more than once) were obtained, compiling into a suffix-compressed packed trie occupying 5943 locations, with 181 outputs. As shown in the table, the resulting algorithm finds 89.3% of the hyphens in the dictionary. This improves on the one and two level examples discussed above. The patterns were generated in 109 passes through the dictionary, requiring about 1 hour of CPU time. 

Examples 

The complete list of hyphenation patterns currently used by TfjjX82 appears in the appendix. The digits appearing between the letters of a pattern indicate the hyphenation level, as discussed above. 

Below we give some examples of the patterns in action. For each of the following words, we show the patterns that apply, the resulting hyphenation values, and the hyphenation obtained. Note that if more than one hyphenation value is specified for a given intercharacter position, then the higher value takes priority, in accordance with our level scheme. If the final value is odd, the position is an allowable hyphen point. 

computer 4mlp pu2t Spute put3er Co4m5pu2t3er com-put-er algorithm Ilg4 Igo3 lgo 2ith 4hm allg4o3r2it4hm al-go-rithm hyphenation hy3ph he2n hena4 hen5at lna n2at itio 2io hy3phe2n5a4t2ion hy-phen-ation 

concatenation o2n onlc lea lna n2at Itio 2io 

co2nlcateln2alt2ion con-cate-na-tion 

mathematics math3 ath5em th2e lma atli c 4cs 

math5eimatli4cs math-e-mat-ics  
**38 PATTERN GENERATION** 

**typesetting type3 els2e 4t3t2 2tlin type3s2e4t3t2ing** 

**type-set-ting** 

**program pr2 lgr pr2olgram pro-gram** 

**supercalifragilisticexpialidocious** 

**ulpe rlc ica alii agli gil4 ilii il4iet islti st2i sltic lexp x3p pi3a 2ila i2al 2id ldo lei 2io 2UB** 

**8ulperlcallifraglil4islt2iclex3p2i3al2ildolc2io2uB** 

**su-per-cal-ifrag-ilis-tic-ex-pi-ali-do-cioua** 

**"Below, we show a few interesting patterns. The reader may like to try figuring out what words they apply to. (The answers appear in the Appendix.)** 

**`ainbo`** `aySal earSk e2mel`   
`hach4 hEelo if4fr IBogo`   
**`n3uin`** 

**`nyp4`** 

**`oSaSles orew4`**   
**`Bspai 4tarc 4todo uir4m`** 

**And finally, the following patterns deserve mention: 3tex fon4t highS**  
*Chapter 5* 

**History and Conclusion** 

\-The invention of the alphabet was one of the greatest advances in the history of civilization. However, the ancient Phoenicians probably did not anticipate **the** fact that, centuries later, the problem of word hyphenation would become a major headache for computer typesetters all over the world. 

Most cultures have evolved a linear style of communication, whereby a train of thought is converted into a sequence of symbols, which are then laid out in **neat** rows on a page and shipped off to a laser printer. 

The trouble was, as civilization progressed and words got longer and longer, it became occasionally necessary to split them across lines. At first hyphens were inserted at arbitrary places, but in order to avoid distracting breaks such as the rapist, it was soon found preferable to divide words at syllable boundaries. 

Modern practice is somewhat stricter, avoiding hyphenations that might cause the reader to pronounce a word incorrectly (e.g. eoneidera-tion) or where a single letter is split from a component of a compound word (e.g. cardi-ovascular). 

The first book on typesetting, Joseph Moxon's *Mechanick Exercises* (1683), mentions the need for hyphenation but does not give any rules for it. A few dictio naries had appeared by this time, but were usually just word lists. Eventually they began to show syllable divisions to aid in pronunciation, as well as hyphenation. 

With the advent of computer typesetting, interest in the problem was renewed. Hyphenation is the \*H' of 'H *\&i* J' (hyphenation and justification), which are the basic functions provided by any typesetting system. The need for automatic hy phenation presented a new and challenging problem to early systems designers. 

Probably the first work on this problem, as well as many other aspects of com puter typesetting, was done in the early 1950s by a French group led by G. D. Bafour. They developed a hyphenation algorithm for French, which was later adapted to English \[U.S. Patent 2,702,485 (1955)\]. 

Their method is quite simple. Hyphenations are allowed anywhere in a word except among the following letter combinations: before two consonants, two vawcJs, 

**39**  
**40 HISTORY AND CONCLUSION** 

or x; between two vowels, consonant-h, e-r, or s-s; after two consonants **where the** first is not 1, m, n, r, or s; or after c, j , q, v, consonant-w, nra, lr, nb, nf, nl, nm, nn, or nr. 

We tested this method on our pocket dictionary, and it found nearly 70 percent of the hyphens, but also about an equal amount of incorrect hyphens\! Viewed in another way, about 65% of the erroneous hyphen positions are successfully inhibited, along with 30% of the correct hyphens. It turns out that a simple algorithm like this one works quite well in French; however for English this is not the case. 

Other early work on automatic hyphenation is described in the proceedings of various conferences on computer typesetting (e.g. \[30\]). A good summary appears in \[31\], from which the quotes in the following paragraphs were taken. 

At the Los Angeles Times, a sophisticated logical routine was developed based on the grammatical rules given in Webster's, carefully refined and adapted for com puter implementation. Words were analyzed into vowel and consonant patterns which were classified into one of four types, and rules governing each type applied. Prefix, suffix, and other special case rules were also used. The results were report edly "85-95 percent accurate", while the hyphenation logic occupies "only 5,000 positions of the 20,000 positions of the computer's magnetic core memory, less space than would be required to store 500 8-letter words averaging two hyphens per word." 

Perry Publications in Florida developed a dictionary look-up method, along with their own dictionary. An in-corc table mapped each word, depending on its first two letters, into a particular block of words on tape. For speed, the dictionary was divided between four tape units, and "since the RCA 301 can search tape in both directions," each tape drive maintained a "homing position" at the middle of the tape, with the most frequently searched blocks placed closest to the homing positions. 

In addition, they observed that many words could be hyphenated after the 3rd, 5th, or 7th letters. So they removed all such words from the dictionary (saving some space), and if a word was not found in the dictionary, it was hyphenated after the 3rd, 5th, or 7th letter. 

A hybrid approach was developed at the Oklahoma Publishing Company. First some logical analysis was used to determine the number of syllables, and to check if certain suffix and special case rules could be applied. Next the probability of hyphenation at each position in the word was estimated using three probability tables, and the most probable breakpoints were identified. (This seems to be the origin of the Time magazine algorithm described in Chapter 1.) An exception  
**HISTORY AND CONCLUSION 41** 

**dictionary handles the remaining cases; however there was some difference of opinion as to the size of the dictionary required to obtain satisfactory results. Many other projects to develop hyphenation algorithms have remained pro prietary or were never published. For example, IBM alone worked on "over 35 approaches to the simple problem of grammatical word division and hyphenation". By now, we might have hoped that an "industry standard" hyphenation algo rithm would exist. Indeed Berg's survey of computerized typesetting \[32\] contains a description of what could be considered a "generic" rule-based hyphenation algo rithm (he doesn't say where it comes from). However, we have seen that any logical routine must stop short of complete hyphenation, because of the generally illogical basis of English word division.** 

**The trend in modern systems has been toward the hybrid approach, where a logical routine is supplemented by an extensive exception dictionary. Thus the in core algorithm serves to reduce the size of the dictionary, as well as the frequency of accessing it, as much as possible.** 

**A number of hyphenation algorithms have also appeared in the computer sci ence literature. A very simple algorithm is described by Rich and Stone \[33\]. The two parts of the word must include a vowel, not counting a final e, ee or ed. The new line cannot begin with a vowel or double consonant. No break is made between the letter pairs eh, gh, p, ch, th, wh, gr, pr, cr, tr, wr, br, f r, dr, vowel-r, vowel-n, or om. On our pocket dictionary, this method found about 70% of the hyphens with 45% error.** 

**The algorithm used in the Bell Labs document compiler Roff is described by Wagner \[34\], It uses suflix stripping, followed by digram analysis carried out in a back to front manner. In addition a more complicated scheme is described using four classes of digrams combined with an attempt to identify accented and nonaccented syllables, but this seemed to introduce too many errors. A version of the algorithm is described in \[35\]; interestingly, this reference uses the terms "hyphenating pattern" (referring to a Snobol string-matching pattern) as well as "inhibiting suffix".** 

**Ockcr \[36\], in a master's thesis, describes another algorithm based on the rules in Webster's dictionary. It includes recognition of prcGxes, suffixes, and special letter combinations that help in determining accentuation, followed by an analysis of the "liquidity" of letter pairs to find the character pair corresponding to the greatest interruption of spoken sound.** 

**Moitra et al \[37\] use an exception table, prefixes, suffixes, and a probabilistic break-value table, In addition they extend the usual notion of affixes to any letter**  
42 HISTORY AND CONCLUSION 

pattern that helps in hyphenation, including 'root words' (e.g. **lint, pot)** intended to handle compound words. 

Patterns as paradigm 

Our pattern matching approach to hyphenation is interesting **for** a number of reasons. It has proved to be very effective and also very appropriate for the problem. In addition, since the patterns are generated from the dictionary, it is easy to accommodate changes to the word list, as our hyphenation preferences change or as new words are added. More significantly, the pattern scheme can be readily applied to different languages, if we have a hyphenated word list for the language. 

The effectiveness of pattern matching suggests that this paradigm may be use ful in other applications as well. Indeed more general patten^ matching systems and the related notions of production systems and augmented transition networks (ATN's) are often used in artificial intelligence applications, especially natural lan guage processing. While AI programs try to understand sentences by analyzing word patterns, we try to hyphenate words by analyzing letter patterns. 

One simple extension of patterns that we have not considered is the idea of character groups such as vowels and consonants, as used by nearly all other algo rithmic approaches to hyphenation. This may seem like a serious omission, because a potentially useful meta-pattcrn like 'vowel-consonant-consonant-vowel1 would then expand to 6 x 20 X 20 x C \= 14400 patterns. However, it turns out that a suffix compressed trie will reduce this to just 6 *\+* 20 \+ 20 \+ 6 \= 52 trie nodes. So our methods can take some advantage of such "mcta-patterns". 

In addition, the use of inhibiting as well as hyphenating patterns seems quite powerful. These can be thought of as rules and exceptions, which is another common AI paradigm. 

Concerning related work in AI, we must especially mention the Meta-DENDRAL program \[38\], which is designed to infer automatically rules for mass-spectrometry. An example of such a rule is N—C—C—C —» N—C \* C—C, which says that if the molecular substructure on the left side is present, then a bond fragmentation may occur as indicated on the right side. Meta-DENDRAL analyzes a set of mass-spectral data points and tries to infer a set of fragmentation rules that can correctly predict the spectra of new molecules. The inference process starts with some fairly general rules and then refines them as necessary, using the experimental data as positive or negative evidence for the correctness of a rule.  
HISTORY AND CONCLUSION 43 

The fragmentation rules can in general be considerably more complicated than our simple pattern rules for hyphenation. The molecular "pattern" can be a tree like or even cyclic structure, and there may be multiple fragmentations, possibly involving "migration" of a few atoms from one fragment to another. Furthermore, there are usually extra constraints on the form of rules, both to constrain the search and to make it more likely that meaningful or "interesting" rules will be generated. Still, there arc some striking similarities between these ideas and our pattern-matching approach to hyphenation. 

Packed tries 

Finally, the idea of packed tries deserves further investigation. An indexed trie can be viewed as a finite-state machine, where state transitions are performed by address calculation based on the current state and input character. This is extremely fast on most computers. 

However indexing usually incurs a substantial space penalty because of space reserved for pointers that are not used. Our packing technique, using the idea of storing the index character to distinguish transitions belonging to different states, combines the best features of both the linked and indexed representations, namely space and speed. We believe this is a fundamental idea. 

There are variou" issues to be explored here. Some analysis of different packing methods would be interesting, especially for the handling of dynamic updates to a packed trie. 

Our hyphenation trie extends a finite-state machine with its hyphenation "ac tions". It would be interesting to consider other applications that can be handled by extending the basic finite-state framework, while maintaining as much of its speed as possible. 

Another possibly interesting question concerns the size of the character and pointer fields in trie transitions. In our hyphenation trie half of the space is occupied by the pointers, while in our spelling checking examples from one-half to three fourths of the space is used for pointers, depending on the size of the dictionary. In the latter case it might be better to use a larger "character" size in the trie, in order to get a better balance between pointers and data. 

When performing a search in a packed trie, following links will likely make us jump around in the trie in a somewhat random manner. This can be a disadvantage, both because of the need for large pointers, and also because of the lack of locality, which could degrade performance in a virtual memory environment. There are probably ways to improve on this. For example, Frcdkin \[10\] proposes an interesting 'n-dimcnsional binary trie\* idea for reducing pointer size.  
**44 . HISTORY AND CONCLUSION** 

We **have** presented packed tries as a solution to the **set representation problem,** with special emphasis on data compression. It would be interesting **to** compare **our** results with other compression techniques, such as Huffman coding. Also, perhaps one could estimate the amount of information present in **a** hyphenated word list, **as a** lower bound on the size of any hyphenation algorithm. 

Finally, our view of finite-state machines has been based on **the** underlying assumption of **a** computer with random-access memory. Addressing by indexing seems to provide power not available in some other models of computation, such as pointer machine*,* or comparison-based models. On the other hand, **a** 'VLSI\* or other hardware model (such as programmed logic arrays) can provide even greater power, eliminating the need for our perhaps contrived packing technique. But then other communication issues will be raised. 

**If all** problems **of** hyphenation **have not been solved,** 

**at** least some **progress has been made since that night,** 

when according to legend, **an RCA Marketing Manager** 

received a phone call **from a disturbed customer.** 

His 301 **had Just hyphenated "God".** 

— **Paul E. Justus (1972)**  
TgX82 hyphenation **patterns** 

**`.ach4`**   
**`.ad4d«r ••fit`**   
**`.«13t`**   
**`.••Sit`**   
**`.anSc`**   
**`• »ng4`**   
**`.aniEa`**   
**`.ant4`**   
**`.anSt*`**   
**`.antiSt .arBa`**   
**`.arm* .ar4ty`**   
**`.a*3c`**   
**`.aalp`**   
**`.ailt`**   
**`.««ter5 .itc«5`** 

**`.auld`**   
**`.ar4i`**   
**`• awn4`**   
**`• bi4g`**   
**`.baEna`**   
**`.ba*4«`**   
**`.bar 4`**   
**`.beSra`**   
**`.baSaa`**   
**`.be5fto .brl2`**   
**`.but4tl .caa4pa .canSc`** 

**`.capaSb .carSol .ca4t`**   
**`.ce41a`**   
**`ch4`**   
**`.chillSi .el9`**   
**`.citSr`**   
**`.co3»`**   
**`.co4r`**   
**`.coiSner .de4moi . de3o`** 

**`.de3ra`**   
**`.deSrl`**   
**`.dei4c`**   
**`.dlctloS .do4t`** 

**`.du4c`**   
**`. dumbS .earthB .eat31`** 

**`.eb4`**   
**`.aer4`**   
**`• eg2`**   
**`.el5d`**   
**`.el3e«`**   
**`.enam3`**   
**`• en3g`** 

**`J`**   
**`.«nS»`**   
**`.eqSuiBt .•r4ri`** 

**`.Ml`**   
**`.euS`**   
**`.eyeS`**   
**`.fei3`**   
**`.ior5»ar •g»2`**   
**`.ga2`**   
**`.gen3t4 •geBog`**   
**`.gi5a`**   
**`.gl4b`**   
**`• go4r`**   
**`.handBi .hanBk`**   
**`.he 2`**   
**`.heroSi .hatS`** 

**`.hatS`**   
**`.M3b`**   
**`.hi3er`**   
**`.honSey .hon3o`**   
**`.hoT5`**   
**`.Id41`**   
**`.idolS`**   
**`.ia3a`**   
**`.imSpin .inl`** 

**`.In3cl`**   
**`.ine2`**   
**`.In2k`**   
**`.inSt`**   
**`.ir5r`**   
**`.ii4i`**   
**`.]u3r`**   
**`.Is4cy`**   
**`.l»4a`**   
**`.latSar .lathB`** 

**`.Ie2`**   
**`.legSa`**   
**`.Ien4`**   
**`.lepS`**   
**`.lev!`**   
**`.H4g`**   
**`.ligSa`**   
**`.Ii2n`**   
**`.1130`**   
**`.114t`**   
**`.aagSaS .nalSo`**   
**`.nanSa`**   
**`.narEti .ae2`** 

**`.ner3e`**   
**`•EeEtor .•ill`**   
**`.•lstSl .Bon3a`** 

**`.•oSre`**   
**`.auBta`**   
**`.•utaSb`**   
**`.nl4e`**   
**`.od2`**   
**`.oddS`**   
**`.of5t«`**   
**`.orSato`**   
**`• or 3c`**   
**`.orld`**   
**`.or St`**   
**`.0*3`**   
**`.o»4tl`**   
**`.othS`**   
**`.outS`**   
**`.pud5»l`**   
**`.paBta`**   
**`.pe5t.it .pl4«`** 

**`.pioBn`**   
**`.pi2t`**   
**`.praSa`**   
**`.ra4c`**   
**`.ran4t`**   
**`.ratioBna .rea2`** 

**`.reEait`**   
**`.re«2`**   
**`.raBatat .ri4g`**   
**`,rit5u`**   
**`.ro4q`**   
**`.rosSt`**   
**`.rowSd`**   
**`.ru4d`**   
**`.•ci3a`**   
**`.aelfS`**   
**`.•ellS`**   
**`.•e2n`**   
**`.•eErla`**   
**`.«h2`**   
**`.•12`**   
**`.iing4`**   
**`.•t4`**   
**`.•taSbl`** 

**`•y2`**   
**`.ta4`**   
**`.ta2`**   
**`.tenSan`**   
**`.th2`**   
**`.ti2`**   
**`.til4`**   
**`.tiaSoS`**   
**`.ting4`**   
**`.tinSk`**   
**`,ton4a`**   
**`.to4p`**   
**`.topSl`**   
**`.tonSa`**   
**`.tribBat .vnla`** 

**`.nn3ca`**   
**`«/`**   
**`.cnderS .nnla`**   
**`.nn5k`**   
**`.nn5o`**   
**`• nnSn`**   
**`.op 3`**   
**`.nraS`**   
**`.u»5a`**   
**`.Ten4da .YeEra`** 

**`.wllBl`**   
**`.ya4`**   
**`4ab.`**   
**`aBbal`**   
**`aSban`**   
**`aba2`**   
**`abBard`**   
**`abiSa`**   
**`abSitSab abSlat`**   
**`abSoSlli 4abr`**   
**`abBrog`**   
**`ab3ul`**   
**`a4car`**   
**`acSard`**   
**`acSaro`**   
**`a5ceon`**   
**`aclar`**   
**`aSchat`**   
**`4a2ci`**   
**`a3cie`**   
**`aclin`**   
**`a3cio`**   
**`ac5rob`**   
**`act5if`**   
**`ac3ul`**   
**`ac4um`**   
**`a2d`**   
**`ad4din`**   
**`adSar.`**   
**`2adi`**   
**`a3dla`**   
**`ad3ica`**   
**`adl4er`**   
**`a3dio`**   
**`a3dit`**   
**`aSdiu`**   
**`ad41e`**   
**`ad3cv`**   
**`adSran`**   
**`ad4su`**   
**`4adu`**   
**`a3duc`**   
**`adSum`**   
**`ae4r`**   
**`aeri4a`**   
**`a2f`**   
**`aff4`**   
**`a4gab`**   
**`aga4n`**   
**`aB 5ell`**   
**`>/`**   
**`aga4o`**   
**`4ageu`** 

**`•8li`**   
**`4ag41`**   
**`agin`**   
**`a2go`**   
**`Sagog`**   
**`ag3onl aSguar agBnl`** 

**`•<8T`**   
**`a3ha`**   
**`a3ha`**   
**`ah 41`**   
**`»3ho`**   
**`a 12`**   
**`aSla`**   
**`•Sic.`**   
**`tiSly`**   
**`a414n`**   
**`ainSin ainBo`** 

**`aitSen`** `•1J`   
**`aklan`**   
**`alBab`**   
**`al3ad`**   
**`a41ar`**   
**`4aldi`**   
**`2ala`**   
**`al3end a41entl a51e5o alii`** 

**`al4ia. ali4e`**   
**`al51er 4allic 4alB`**   
**`a51og. a41y.`**   
**`4aly«`**   
**`5a51y«t Salyt`**   
**`3alyi`**   
**`4aaa`**   
**`aaSab`**   
**`am3ag`**   
**`aaaSra araSaec a4matlt`** 

**`a4mSato anBera ai»3ic`** 

**`amSif`**   
**`amSilf ami in`** 

**`ami4no a2ao`** 

**`aBaon`**   
**`amorBi amp5en J`** 

**74**

**`•2n`**   
**`anSag* 3an»ly`** 

**`aSnar`**   
**`anSare`**   
**`anar4i`**   
**`aSnatl`**   
**`4 and`**   
**`ande4»`**   
**`an3di»`**   
**`anldl`**   
**`an4dov`**   
**`aSnas`**   
**`aSnan`**   
**`an5e«t. »3n«u`** 

**`2ang`**   
**`angBl* anlgl`** 

**`a4nlic`**   
**`a3niai`**   
**`an313f`**   
**`an4iaa`**   
**`aSnlai a5nina`** 

**`anSio`**   
**`a3nip`**   
**`an3iih anSit`**   
**`a3nin`**   
**`an4kll`**   
**`Sannix`**   
**`anoi`**   
**`anSot`**   
**`anothS`**   
**`an2sa`**   
**`an4sco`**   
**`an4an`**   
**`an2«p`**   
**`ans3po`**   
**`an4tt`**   
**`an4inr`**   
**`antal4 an4tie`** 

**`4 an to`**   
**`an2tr`**   
**`an4tw`**   
**`an3ua`**   
**`an3ul`**   
**`aSnur`**   
**`4ao`**   
**`apar4`**   
**`apfiat`**   
**`apSero`**   
**`a3phar`**   
**`4aphi`**   
**`a4pilla apSillar ap3in`** 

**`ap3ita`**   
**`a3pitu`**   
**`a2pl`**   
`y`   
**`•pocB`**   
**`apSola`**   
**`aporEl`**   
**`apotSt`**   
**`apiSet`**   
**`•Spo`**   
**`•queS`**   
**`2a2r`**   
**`arSact`**   
**`aSrad*`**   
**`arBadlt •rSal`**   
**`aSraaats aran4g`** 

**`ara3p`**   
**`ar4at`**   
**`aBratie arSatlr aBran`** 

**`arBaT4`**   
**`•raw4`**   
**`arbal4`**   
**`ar4cha« arSdina ar4dr`** 

**`arBaaa`**   
**`aSraa`**   
**`arSant`**   
**`aSraas`**   
**`ar4fl`**   
**`ar4M`**   
**`aril`**   
**`arSlal`**   
**`arSian`**   
**`a3riat`**   
**`ar4ia`**   
**`arSinkb ar31o`**   
**`ar2ix`**   
**`ar2al`**   
**`arBoSd`**   
**`aBroni`**   
**`a3roo`**   
**`ar2p`**   
**`arSq`**   
**`arre4`**   
**`ar4ta`**   
**`ar2ih`**   
**`4aa.`**   
**`aa4ab`**   
**`»3ant`**   
**`aahl4`**   
**`a5iia.`**   
**`aSalb`**   
**`a3«lc`**   
**`EaEai4t atk31`** 

**`••41`**   
**`a4aoc`**   
**`aaSph`**   
**`aa4fh`**   
**`aiStea`**   
**`J`**   
**`atltr`**   
**`iiurBa`**   
**`•2ta`**   
**`at-Sabl`**   
**`•tSae`**   
**`atSalo`**   
**`•tSap`**   
**`ataSc`**   
**`atBaek`**   
**`atSago`**   
**`atSan.`**   
**`atSara`**   
**`aterBn`**   
**`aBtarna atSaat`** 

**`atBar`**   
**`4»th`**   
**`athEea`**   
**`aEthaa`**   
**`at4ho`**   
**`athSoa`**   
**`4ati.`**   
**`aStla`**   
**`atSlSa`**   
**`atllc`**   
**`•tsit`**   
**`atlonSar atSltu`** 

**`a4tog`**   
**`a2toa`**   
**`atSoali a4top`**   
**`»4to»`**   
**`altr`**   
**`atBrop`**   
**`at4ak`**   
**`aUtag`**   
**`atSt*`**   
**`at4th`**   
**`a2tn`**   
**`atSua`**   
**`atSua`**   
**`•t3nl`**   
**`at3ura`**   
**`a2ty`**   
**`au4b`**   
**`anghS`**   
**`au3gu`**   
**`au412`**   
**`aunSd`**   
**`au3r`**   
**`auBalb`**   
**`ait&en`**   
**`uulth`**   
**`a2va`**   
**`av3ag`**   
**`•Sran`**   
**`»ve4no`**   
**`sv3era`**   
**`arSera`**   
**`arSery`**   
**`aril`**   
**`J`**   
**`a»14ar`**   
**`aySlg`**   
**`aySoe`**   
**`alTor`**   
**`3avaj`**   
**`aw3i`**   
**`a«41y`**   
**`avi4`**   
**`ax4ie`**   
**`ax4id`**   
**`aySal`**   
**`aT«4`**   
**`ay.4`**   
**`axl4«r`**   
**`axxSl`**   
**`Eba.`**   
**`badSgar ba4ge`** 

**`balla`**   
**`banSdag ban4«`** 

**`banSl`**   
**`barblB`**   
**`bari4a`**   
**`fci«4fi lbat`**   
**`b*4i`**   
**`2blb`**   
**`b2ba`**   
**`bSbar`**   
**`bbUna`**   
**`4bld`**   
**`4ba.`**   
**`baak4`**   
**`boats`**   
**`4ba2d`**   
**`baSda`**   
**`ba3d*`**   
**`bo3di`**   
**`bajgl`**   
**`baSgn`**   
**`lbal`**   
**`belli`**   
**`baSlo`**   
**`4be5»`**   
**`baSnlg`**   
**`baEnn`**   
**`4be»4`**   
**`bo3ap`**   
**`beSctr`**   
**`3bet`**   
**`betSii`**   
**`baStr`**   
**`ba3tw`**   
**`be3w`**   
**`beSyo`**   
**`2bf`**   
**`4b3h`**   
**`bl2b`**   
**`bi4d`**   
**`3bla`**   
**`blEan`**   
`•J`   
**TgX82** HYPHENATION **PATTERNS 75** 

**`bi4ar`**   
**`2b31f`**   
**`lbll`**   
**`bl311i bim5r4 bln4d`** 

**`blSnet bi3ogr bi6oa`** 

**`bi2t`**   
**`3bl3tio bl3tr`** 

**`3blt5M bSltt`** 

**`blj`**   
**`bk4`**   
**`b212`**   
**`blithS b41a.`** 

**`blan4`**   
**`Bbleip I31ia`**   
**`b41o`**   
**`blnn4t 4bla`**   
**`4b3n`**   
**`bneSg`**   
**`Sbod`**   
**`bod31`**   
**`bo4a`**   
**`bolSlc`**   
**`boMbl`**   
**`bon4i`**   
**`bonSat Sboo`** 

**`6bor.`**   
**`4blora`**   
**`borSd`**   
**`Ebor*`**   
**`6bjrl`**   
**`6toi4`**   
**`b5ot»`**   
**`bothS`**   
**`bo4to`**   
**`bounds 4bp`** 

**`4brlt`**   
**`brothS 2bSt2`** 

**`bior4`**   
**`2bt`**   
**`bt41`**   
**`b4to`**   
**`b3tr`**   
**`buf4f»r bu4ga`** 

**`buSU`**   
**`buaU`**   
**`bu4n`**   
**`buntii fcu3ra`**   
**`bntSl* b«ti4« SV«at`**   
**`4b«U`**   
**`Sbati*`**   
**`bSuto`**   
**`blr`**   
**`4b5»`**   
**`6 by.`**   
**`bya4`**   
**`lea`**   
**`cab3ia calbl`**   
**`cach4`**   
**`*c»5den 4cag4`**   
**`2c5ah`**   
**`ca31at cal4U`** 

**`callSU 4calo`** 

**`canEd`**   
**`can4«`**   
**`can41c`**   
**`canSla`**   
**`can31z c»n4ty`** 

**`cany4`**   
**`caSpar`**   
**`carEoa`**   
**`cattSer caiStlg 4c*iy`** 

**`ca4U`**   
**`4catiT`**   
**`C«T5«1`**   
**`cSc`**   
**`cchaB`**   
**`ccl4a`**   
**`ccoapat ccon4`** 

**`ecouSt`**   
**`8c*.`**   
**`4c*d.`**   
**`4cadaa Seal`** 

**`6c«l.`**   
**`Scall`**   
**`lean`**   
**`Scane`**   
**`2can4« 4c«nl`** 

**`Scant`**   
**`Scap`**   
**`coSrim`**   
**`4cata`**   
**`Scaial`**   
**`ce«5»i6b cat5t`**   
**`cat4`**   
**`c6a4U`**   
**`ca»4`**   
**`3ch`**   
**`4ch.`**   
**`4ch3ab`**   
**`Echanle chEaSnla cl.2`** 

**`CAM» 9`**   
**`4cka4`**   
**`(USU`**   
**`y`** 

**`3che«i`**   
**`chBana`**   
**`ch3e-.`**   
**`ch3or«`**   
**`4cklln`**   
**`Schina. chSlnaaa Echini`** 

**`Echlo`**   
**`Schlt`**   
**`chl2z`**   
**`3cho2`**   
**`ch4ti`**   
**`lei`**   
**`Scia`**   
**`ci2aRb`**   
**`cla6r`**   
**`ci6c`**   
**`4ciar`**   
**`Eclfie. 4cli`** 

**`ci41a`**   
**`Scili`**   
**`2cla`**   
**`2cin`**   
**`e41na`**   
**`Scinat`**   
**`cln3ea`**   
**`cling`**   
**`cElng.`**   
**`Sclno`**   
**`clon4`**   
**`4clp«`**   
**`clSpK`**   
**`4clplc`**   
**`4claU`**   
**`4clatl`**   
**`2c lit`**   
**`CltSll`**   
**`Sell`**   
**`ckl`**   
**`ckSl`**   
**`Ic414`**   
**`4clar`**   
**`cSlaratl* Eclar*`** 

**`cla4a`**   
**`4cllc`**   
**`clla4`**   
**`cly4`**   
**`cEn`**   
**`lco`**   
**`coSag`**   
**`coa2`**   
**`2cog`**   
**`co4gr`**   
**`col4`**   
**`coSinc`**   
**`colfil`**   
**`Ecole`**   
**`colScr`**   
**`co»5«r`**   
**`co»4a`**   
**`c4ea«`**   
**`cokSg`**   
**`co*5V`**   
`J` 

**`co3p»`**   
**`cop3ic co4pl`** 

**`4corb`**   
**`coro3a co«4»`**   
**`corl`**   
**`C0T64`**   
**`covEa`**   
**`cozfi*`**   
**`coExl`**   
**`clq`**   
**`craaSt ficrat. Ecratlc cre3at Scred`** 

**`4c3reU cr«4r`** 

**`cri2`**   
**`crlSf`**   
**`c4rln`**   
**`crii4`**   
**`Scrltl cro4pl crop5o cro«4e crc4d`**   
**`4c3»2`**   
**`2clt`**   
**`Cta4b`**   
**`ctSang c6tant c2t«`** 

**`c3t«r`**   
**`c4tico ctia3i ctu4r`** 

**`c4tw`**   
**`cudS`**   
**`c4uf`**   
**`c4ul`**   
**`cu5ity Bculi`**   
**`cul4tla Scultu cu2aa`** 

**`cSuaa`**   
**`cu4al`**   
**`Scun`**   
**`cu3pl`**   
**`cu5py`**   
**`carSa4b cuSrla lena`** 

**`cuaa41 3c4ut`**   
**`cu4ti« 4c5utiT 4cutr`**   
**`ley`**   
**`cie4`**   
**`Id2a`**   
**`Ma.`**   
**`2d3a4b dach4`** 

**`J`** 

**`4daf`**   
**`2dag`**   
**`da2a2`**   
**`dan3g`**   
**`dardS`**   
**`darkS`**   
**`4dary`**   
**`3dat`**   
**`4dat.iT 4dato`**   
**`6dtT4`**   
**`daTE*`**   
**`6d*y`**   
**`dlb`**   
**`dSc`**   
**`dld4`**   
**`2de.`**   
**`deafE`**   
**`deb&it do4bon decan4 do4cil do5coa 2diod`**   
**`4doo.`**   
**`do5if`**   
**`deli4e do!5iSq doSio`**   
**`d4co`**   
**`Edea.`**   
**`3dooic deaSic. deSall de4aona deaorS lden`**   
**`do4mr d«3no`**   
**`dantlEf do3nu`** 

**`delp`**   
**`do3p«`**   
**`depl4`**   
**`de2pu`**   
**`d3aq`**   
**`d4«rh`**   
**`Edara`**   
**`dernEli der&a`**   
**`daa2`**   
**`d2aa.`**   
**`delac`**   
**`de2tSo de*3tl de3atr do4«tt`** 

**`delt`**   
**`da2to`**   
**`d«lr`**   
**`dov3il 4dey`** 

**`4dlt`**   
**`d4g»`**   
**`d3ge^t djll`** 

**`d2gy`**   
**`dlh2`**   
**`Edi.`**   
**`•d413a`**   
**`diafib`**   
**`di4caa d4ice`**   
**`3dict`**   
**`3did`**   
**`5di3on`**   
**`dllf`**   
**`di3ga`**   
**`di41ato dlln`** 

**`ldina`**   
**`Sdina. Edinl`**   
**`dlEnlx ldio`** 

**`dioEg`**   
**`di4pl`**   
**`dir2`**   
**`dilra`**   
**`dirtSi disl`** 

**`Edi«i`**   
**`d4is3t d2iti`** 

**`ldllT`**   
**`dlj`**   
**`d5)c2`**   
**`4d51a`**   
**`3dle.`**   
**`3dled`**   
**`3dlaa. 4dloia`** 

**`2d31o`**   
**`4d5lB`**   
**`2dly`**   
**`dla`**   
**`4dln4`**   
**`Ido`**   
**`3do.`**   
**`doEda`**   
**`Edoa`**   
**`2dSof`**   
**`d4og`**   
**`do41a`**   
**`doli4`**   
**`doBlor doa5iz do3nat donl4`** 

**`doo3d`**   
**`dop4p`**   
**`d4or`**   
**`3doa`**   
**`4d5oat`**   
**`do4r`**   
**`3dox`**   
**`dip`**   
**`ldr`**   
**`dragSoa <dr»i`** 

**`dre4`**   
**`dreaSr`** `J` 

**`Edren`**   
**`dri4b`**   
**`dril4`**   
**`dro4p`**   
**`4drow`**   
**`Edrupll 4dry`**   
**`2dla2`**   
**`da4p`**   
**`d4av`**   
**`d4«y`**   
**`d2th`**   
**`Ida`**   
**`dlola`**   
**`du2c`**   
**`dluca`**   
**`dacSer 4duct.`**   
**`4d'icti`**   
**`du5el`**   
**`du4g`**   
**`d3ulo`**   
**`dua4ba du4n`**   
**`4dap`**   
**`du4p«`**   
**`dlT`**   
**`dlw`**   
**`d2y`**   
**`5dyn`**   
**`dy4aa`**   
**`dyaSp`**   
**`ela4b`**   
**`«3act`**   
**`•adl`**   
**`aadSl* •»4g«`**   
**`aaEgar aa41`** 

**`aalEar •I13OB`** 

**`aan3ar aSand`** 

**`aar3a`**   
**`aar4c`**   
**`aarSaa aar41c aar411 aarSk`**   
**`aar2t`**   
**`e»rt3»`**   
**`aaSap`**   
**`e3a»«`**   
**`aaatS`**   
**`•*2t`**   
**`aatfien eath31 aSatlf`**   
**`«4a3tB`**   
**`B»2 T`**   
**`eaT3on`**   
**`aaTSi`**   
**`eaTSo`**   
**`I.Ik`**   
**`•4b«l. •4bel«`**   
**`/`** 

**`e4ben`**   
**`e<bit`**   
**`e3br`**   
**`a4cad`**   
**`«canEc`**   
**`ecca5`**   
**`•lea`**   
**`•cSeaaa`**   
**`ec2i`**   
**`e4cib`**   
**`ecSificat ecSifla`** 

**`ecBlfy`**   
**`acSia`**   
**`•cl4t`**   
**`eSclt*`**   
**`•4claa`**   
**`a4clna`**   
**`a2col`**   
**`a4coaa`**   
**`e4coap«`**   
**`e4conc`**   
**`e2cor`**   
**`acSora`**   
**`ecoSro`**   
**`alcr`**   
**`a4crea`**   
**`ec4ta&`**   
**`ac4t«`**   
**`alca`**   
**`a4cal`**   
**`ec3ula`**   
**`2a2da`**   
**`4ed3d`**   
**`a4dl«r`**   
**`ade4a`**   
**`4adl`**   
**`• 3dia`**   
**`adSib`**   
**`• ed3ica`**   
**`adMa`**   
**`•dlit`**   
**`adlfii`**   
**`4edo`**   
**`•4dol`**   
**`adon2`**   
**`a4drl`**   
**`• 4dul`**   
**`adSule`**   
**`aa2c`**   
**`aed31`**   
**`aa2f`**   
**`aal31`**   
**`««41y`**   
**`aa2a`**   
**`••4na`**   
**`«e4pl`**   
**`aa2a4`**   
**`aatt4`**   
**`eo4ty`**   
**`aSas`**   
**`•If`**   
**`e«f3«ra`**   
**`lalf`**   
**`•4flc`**   
**`Safici`**   
`J` 

**`afil4`**   
**`•3fina`**   
**`atSiSnlt* Satlt`** 

**`•forEa*`**   
**`•4fva«.`**   
**`4agal`**   
**`•gar4`**   
**`•gSlb`**   
**`•g41c`**   
**`•gElng`**   
**`aSgltS`**   
**`agEn`**   
**`a4go.`**   
**`a4go«`**   
**`eglal`**   
**`•Sgur`**   
**`Bagy`**   
**`•lb.4`**   
**`eher4`**   
**`•12`**   
**`•51c`**   
**`•16d`**   
**`•Ig2`**   
**`•lEgl`**   
**`•Slab`**   
**`•3inf`**   
**`•Ung`**   
**`•Elnat`**   
**`•Ir4d`**   
**`• it3a`**   
**`alSth`**   
**`• 6ity`**   
**`•1J`**   
**`•4]nd`**   
**`•jEadl`**   
**`akl4n`**   
**`•k41a`**   
**`alia`**   
**`•41a.`**   
**`•41ac`**   
**`•Ian4d`**   
**`•lEatlT`**   
**`a41av`**   
**`•Iaza4`**   
**`•Sl«a`**   
**`•IBabra`**   
**`Selac`**   
**`•41«d`**   
**`•13aga`**   
**`•Elan`**   
**`•411«r`**   
**`•ll«a`**   
**`•12f`**   
**`•121`**   
**`•31ib«`**   
**`•4161c.`**   
**`•131ca`**   
**`•311rr`**   
**`•161gl»`**   
**`•Ella`**   
**`•413in«`**   
**`•311o`**   
**`•211a`**   
**`•lSlak`** 

**`•311*37`**  
**76 HYPHENATION PATTEON8** 

**`4ella`**   
**`el41ab`**   
**`ello4`**   
**`aSloe`**   
**`•16og`**   
**`•13op.`**   
**`•12«h`**   
**`•14ta`**   
**`•Sln d <• el Bug`** 

**`•4aae`**   
**`•4nag`**   
**`eEnan`**   
**`eaSana`**   
**`eaSb`**   
**`elite`**   
**`•2nal`**   
**`e4met`**   
**`eaSica`**   
**`eal4e`**   
**`emSlgra emlir.2`** 

**`eoSlna`**   
**`em3i3ni •4ai«`**   
**`emSiah`**   
**`•Emiaa`**   
**`•n31x`**   
**`Semnlx`**   
**`eno4g`**   
**`•monlSo •a3pl`** 

**`•4mul`**   
**`emSula`**   
**`omuSn`**   
**`•3my`**   
**`•nSaao`**   
**`•4nant`**   
**`•nch4ar en3dlc`**   
**`eSnaa`**   
**`eBnee`**   
**`•n3ea`**   
**`enBtro`**   
**`•nSeal`**   
**`•nSeat`**   
**`•n3etr`**   
**`•3new`**   
**`•nSlcf`**   
**`•Snia`**   
**`•Enll`**   
**`a3nlo`**   
**`en31»h`**   
**`•n31t`**   
**`t-Snltt`**   
**`Sanil`**   
**`4enn`**   
**`4eno`**   
**`ano4g`**   
**`e4noi`**   
**`an3oT`**   
**`en4aw`**   
**`entbage 4anthaa on3vu`**   
**`•nEuf`** 

**`•3ny.`**   
**`4en3x`**   
**`•Sof`**   
**`•o2g`**   
**`•4ol4`**   
**`•3ol`**   
**`aop3ar`**   
**`alor`**   
**`' eo3re`**   
**`eoSrol`**   
**`eos4`**   
**`•4ot`**   
**`eo4to`**   
**`•Soot`**   
**`•Sow`**   
**`•2pa`**   
**`•3pal`**   
**`epSanc`**   
**`• Spel`**   
**`e3pent`**   
**`epSetitie ephe4.`**   
**`e4pli`**   
**`alpo`**   
**`e4prec`**   
**`.epSreca`**   
**`•4pr«d`**   
**`•p3rah`**   
**`•3pro`**   
**`•4prob`**   
**`•p4ah`**   
**`•pStlEb`**   
**`•4pnt`**   
**`epButa`**   
**`•lq`**   
**`•qulSl`**   
**`e4q3ul3a`**   
**`aria`**   
**`era4b`**   
**`4er»nd`**   
**`•r3ar`**   
**`4er»tl.`**   
**`2erb`**   
**`•r4bl`**   
**`•r3ch`**   
**`•r4cha`**   
**`2«ra.`**   
**`•3real`**   
**`araSco`**   
**`ereSin`**   
**`•rSal.`**   
**`ar3eao`**   
**`arSena`**   
**`•rSenca`**   
**`4«rena`**   
**`arSant`**   
**`ara4q`**   
**`arEaai`**   
**`nrSeat`**   
**`areU`**   
**`arlh`**   
**`aril`**   
**`alrla4`**   
**`Serlck`**   
**`e3rien`**   
**`•rl4«r`** 

`• \J` 

**`or3ine`**   
**`elrio`**   
**`4erlt`**   
**`er41n`**   
**`erl4T`**   
**`e4rira`**   
**`er3m4`**   
**`er4nia`**   
**`4ernlt`**   
**`Eernix`**   
**`«r3no`**   
**`2ero`**   
**`erSob`**   
**`aSroc`**   
**`ero4r`**   
**`erlon`**   
**`aria`**   
**`er3«et`**   
**`ert3er`**   
**`4ertl`**   
**`•r3tw`**   
**`4aru`**   
**`•rn4t`**   
**`Berwau . els4a`**   
**`e4«age. •4sages ••2c`**   
**`•2*ca`**   
**`••Scan`**   
**`e3»cr`**   
**`••Sen`**   
**`elt2e`**   
**`•2>ec`**   
**`eiEecr`**   
**`e«Bone`**   
**`•4aart. a4aerta •4aerra 4eah`**   
**`•3aha`**   
**`aahEan`**   
**`elal`**   
**`e2»ic`**   
**`•2ald`**   
**`••Elden •aSlgna •2aEia`** 

**`e«414n`**   
**`e»U4ta aal4u`**   
**`aSakln`**   
**`et4ai`**   
**`•2tol`**   
**`e»3olu`**   
**`e2aon`**   
**`aaSona`**   
**`elfp`**   
**`ai3par`**   
**`eiSplra ea4pra`** 

**`2e«a`**   
**`e«4al4b e«tan4`** 

**`ea3tlg`**   
**`aaStla`** 

**`v'`** 

**`4ei2to`**   
**`e3iton 2e«tr`** 

**`e5stro estrucS e2(ur`**   
**`esSurr ••4v`** 

**`ata4b`**   
**`eton4d`**   
**`e3teo`**   
**`ethod3 •tlic`**   
**`eEtlda etin4`**   
**`ati4no eStir`**   
**`eStltlo etSitlr 4etn`** 

**`•tSona •3tra`**   
**`•3tro`**   
**`et3ric et5rif`**   
**`«t3rog et5roa et3ua`**   
**`etEya`**   
**`•tSz`**   
**`4m`**   
**`•Sun`**   
**`•3np`**   
**`•u3ro`**   
**`eu>4`**   
**`•ute4`**   
**`•uti61 euEtr`**   
**`eva2pB e2vaa`** 

**`evEaat eSvea`** 

**`•r3ell`**   
**`evel3o eSveng even4i •Tier`** 

**`eBverb elTi`**   
**`•T3id`**   
**`6T141`**   
**`•4Tin`**   
**`«T14 T`**   
**`•STO C`**   
**`•ET U`**   
**`elwa`**   
**`«4wag`**   
**`•Swea`**   
**`o3»h`**   
**`ewllS`**   
**`e»3ing`**   
**`o3wit`**   
**`lexp`**   
**`Seyc`**   
**`Eeya.`**   
**`ey«4`**   
`J` 

**`lfa`**   
**`fa3bl`**   
**`fab3r`**   
**`fa4c«`**   
**`4fag`**   
**`fain4`**   
**`fallSa 4fa4na famEia Efar`**   
**`farSth fa3ta`**   
**`fa3the 4fato`**   
**`faults 4fEb`**   
**`4fd`**   
**`4fe.`**   
**`feas4`**   
**`feath? fe4b`** 

**`4feca`**   
**`Efect`**   
**`2fod`**   
**`fe311`**   
**`f e 4.-00`**   
**`'en2d`**   
**`fend6e ferl`** 

**`5ferr`**   
**`fev4`**   
**`4flf`**   
**`f4fe»`**   
**`f4fie`**   
**`latin. f2fSia f4fly`**   
**`f2fy`**   
**`4fh`**   
**`lfi`**   
**`fi3a`**   
**`2f3ic. 4f3ical f3ican 4flcata f3icen fi3cer fic4i`** 

**`Sflcia Sficla 4flca`**   
**`fi3cn`**   
**`fiSdel fightS f 1151`** 

**`fiUSin 4fUy`** 

**`2fin`**   
**`Efina`**   
**`fin2d5 fi2ne`**   
**`fIin3g fin4n`**   
**`fis4ti f412`** 

**`fSless J`** 

**`flin4`**   
**`flo3re f21rS`** 

**`4fa`**   
**`4fn`**   
**`lfo`**   
**`Efon`**   
**`fon4de fon4t`**   
**`fo2r`**   
**`foErat forSay foreEt for41`**   
**`fortSa fosS`** 

**`4fEp`**   
**`fra4t`**   
**`fSrea`**   
**`fresSc fri2`** 

**`fril4`**   
**`frolS`**   
**`2f3a`**   
**`2ft`**   
**`?4to`**   
**`f2ty`**   
**`3fu`**   
**`fuEel`**   
**`4fug`**   
**`fu4o>in fuSne`**   
**`fu3rl`**   
**`fusl4`**   
**`fu«4«`**   
**`4futa`**   
**`lfy`**   
**`lga`**   
**`gaf4`**   
**`Sgal.`**   
**`3gall`**   
**`ga31o`**   
**`2gaa`**   
**`gaSmet gSano`** 

**`ganSia ga3nii ganlSxa 4gano`** 

**`gar6n4 ga*«4`**   
**`gath3`**   
**`4gatlr 4gaz`** 

**`g3b`**   
**`gd4`**   
**`2ga.`**   
**`2ged`**   
**`geez4`**   
**`gel41n geSUa ge5liz 4gely`** 

**`lgen`**   
**`ge4nat geSnia`** 

**`4geno`**   
**`4geny`**   
**`lgeo`**   
**`ge3om`**   
**`g4ery`**   
**`Egeal`**   
**`gethS`**   
**`4geto`**   
**`ge4ty`**   
**`ga4T`**   
**`4glg2`** 

**`g3ger`**   
**`ggln6`** 

**`gh31n`**   
**`ghSout gh4to`** 

**`Sgl.`**   
**`Igi4a`**   
**`glaEr`**   
**`gllc`**   
**`Eglcla g4ico`** 

**`gienS`**   
**`Egiea. gil4`** 

**`g3imen 3g4in. glnSga 6g41n« 6gio`**   
**`Sglr`**   
**`gir41`**   
**`g3ial`**   
**`gl4u`**   
**`BgiT`**   
**`3gll`**   
**`gl2`**   
**`gla4`**   
**`gladSl 6glaa`** 

**`lgla`**   
**`gli4b`**   
**`g31ig`**   
**`3glo`**   
**`glo3r`** 

**`gl»`** 

**`gn4a`**   
**`g4na.`**   
**`gnet4t glnl`**   
**`g2nln`**   
**`g4nlo`**   
**`glno`**   
**`g4non`**   
**`lgo`**   
**`3go.`**   
**`gobS`**   
**`6goe`**   
**`3g4o4g go3ia`**   
**`gon2`**   
**`4g3o3na gondoS`** `•y •` 

**`go3nl`**   
**`Bgoo`**   
**`goSrlz`**   
**`gorSou`**   
**`figot.`**   
**`gOTl`** 

**`g3p`**   
**`lgr`**   
**`4grada`**   
**`g4ral`**   
**`gran2`**   
**`figraph. gEraphar EgrapMc 4graphy`**   
**`4gray`**   
**`gr«4n`**   
**`<gree«.`**   
**`4grlt`**   
**`g4ro`**   
**`gruf4`**   
**`8*2`**   
**`gSst*`**   
**`gth3`**   
**`gu4a`**   
**`3guard`**   
**`2gue`**   
**`EgulSt`**   
**`3gun`**   
**`3gna`**   
**`4gu4t`**   
**`g3»`**   
**`lgy`**   
**`2g5y3n`**   
**`gyEra`**   
**`h3ab41`**   
**`h*ch4`**   
**`hae4a`**   
**`hae4t`**   
**`hSagu`**   
**`ha31a`**   
**`halaSa`**   
**`ha'Sa`**   
**`hin4ci`**   
**`han4cy`**   
**`Shand.`**   
**`han4g`**   
**`hangSar`**   
**`hangSo`**   
**`hEa&nlt`**   
**`han4k`**   
**`han4ta`**   
**`hap31`**   
**`hapBt`**   
**`ha3ran`**   
**`haErat`**   
**`h»r2d`**   
**`h»rd3p`**   
**`har41a`**   
**`harpSen`**   
**`harStar`**   
**`haiSs`**   
**`haun4`**   
**`Ehax`**   
**`haz3a`**   
**`Ub`**   
**`V`** 

**`lhead`**   
**`3hear`**   
**`he4can`**   
**`hSecat`**   
**`h4ed`**   
**`heEdoS`**   
**`he3141`**   
**`hel4Ua hel41y hEelo`**   
**`hea4p`**   
**`he2n`**   
**`hena4`**   
**`henSat`**   
**`heoEr`**   
**`hepS`**   
**`h4era`**   
**`hera3p`**   
**`her4ba`**   
**`hereSa h3ern`** 

**`hSero« hSary`** 

**`hlea`**   
**`he2aEp h«4t`** 

**`het4ad hou4`** 

**`hit`**   
**`hlh`**   
**`hiEan`**   
**`hl4co`**   
**`hlghS`**   
**`M112`**   
**`hiBor4`**   
**`h41na`**   
**`hlon4a`**   
**`hl4p`**   
**`hlr41`**   
**`hi3ro`**   
**`hir4p`**   
**`hlr4r`**   
**`hl«3el`**   
**`hl*4a`**   
**`hithSar hl2r`** 

**`4hk`**   
**`4hU4`**   
**`hlan4`**   
**`h21o`**   
**`hlo3rl`**   
**`4 M B`**   
**`hmot4`**   
**`2hln`**   
**`hSodii`**   
**`hSoda`**   
**`ho4g`**   
**`hoge4`**   
**`holSar`**   
**`3hol4a`**   
**`ho4aa`**   
**`hoae3`**   
**`hon4a`**   
**`hoSny`**   
**`3hcod`**   
**`AO0A4`**   
`y` 

f76«  
**HYPHENATION PATTERNS 77** 

**`horfiat`**   
**`hoSrit`**   
**`hortSa`**   
**`hoSro`**   
**`hoi4e`**   
**`hoSsen`**   
**`hoalp`**   
**`Jhou«.`**   
**`hou»e3`**   
**`hor5«l`**   
**`4h5p`**   
**`4hr4`**   
**`hreoS`**   
**`hroEnlt hro3po`** 

**`4hl«2`**   
**`h4ih`**   
**`h4tar`**   
**`htlen`**   
**`htSe«`**   
**`h<ty`**   
**`hu4g`**   
**`hu4min`**   
**`hunSka`**   
**`hun4t`**   
**`hci3t4 ' hu4t`** 

**`hlw`**   
**`h4wart`**   
**`hy3pa`**   
**`hy3ph`**   
**`hy2t`**   
**`211*`**   
**`i2al`**   
**`lam*`**   
**`laaSata 12an`**   
**`4ianc`**   
**`ian3i`**   
**`4ian4t`**   
**`laEp*`**   
**`iaaa4`**   
**`14atir`**   
**`ia4trle i4ata`** 

**`Ibe4`**   
**`ib3era`**   
**`ib5ert`**   
**`ibSia`**   
**`ib3in`**   
**`ibSit.`**   
**`ibBlta`**   
**`ilbl`**   
**`Ib311`**   
**`iEbo`**   
**`llbr`**   
**`12bSri`**   
**`lSbun`**   
**`41cam`**   
**`Eicap`**   
**`41car`**   
**`14car.`**   
**`14cara`**   
**`lcaaS`**   
**`14cay`**   
**`lcc««`** 

**`J •`**   
**`4iceo`**   
**`4ich`**   
**`2icl`**   
**`ificid`**   
**`icSina 12cip`** 

**`icSipa 14cly`**   
**`12cEoe 411cr`**   
**`Eicra`**   
**`i4cry`**   
**`Ic4t«`**   
**`ictu2`**   
**`ic4t3ua ic3ula ic4ua`** 

**`icBuo`**   
**`i3cur`**   
**`21d`**   
**`i4dai`**   
**`idSanc ldSd`** 

**`ide3al ida4a`** 

**`12dl`**   
**`ldSian Idl4ar iSdia`** 

**`id3io`**   
**`idifi i ldlit`** 

**`idElu`**   
**`13dl«`**   
**`14doa`**   
**`id3o*`**   
**`14dr`**   
**`12dn`**   
**`ldSuo`**   
**`21«4`**   
**`ied4«`**   
**`EleSga Iflld3`** 

**`ienEa4 Ien4a`** 

**`iSenn`**   
**`13entl liar.`** 

**`13atc`**   
**`ilaat`**   
**`13et`**   
**`411.`**   
**`ifSaro ltfSan 1141r`**   
**`4111c. 1311a`** 

**`1311`**   
**`41ft`**   
**`2U`**   
**`iga5b`**   
**`ig3era Ight31 41gi`**   
**`13glb`**   
**`1(311`**   
**`J`**   
**`Ig3in`**   
**`ig3it`**   
**`14g41`**   
**`12go`**   
**`Ig3or`**   
**`igSot`**   
**`iBgra`**   
**`iguSi`**   
**`iglnr`**   
**`13h`**   
**`41E14`**   
**`13j`**   
**`41k`**   
**`Ilia`**   
**`113a4b`**   
**`141ada`**   
**`1216aB`**   
**`llaEra`**   
**`131eg`**   
**`lller`**   
**`Iler4`**   
**`1161`**   
**`1111`**   
**`1131a`**   
**`1121b`**   
**`1131o`**   
**`1141st`**   
**`2ilit`**   
**`1121z`**   
**`lllSab`**   
**`411n`**   
**`ilSoq`**   
**`114ty`**   
**`HEur`**   
**`113T`**   
**`14mag`**   
**`Im3aga`**   
**`iraaBry`**   
**`inentaSr 41met`** 

**`lmll`**   
**`lmSida`**   
**`lmlEla`**   
**`lEmlnl`**   
**`41nlt`**   
**`In4nl`**   
**`13oon`**   
**`12mn`**   
**`inSuli`**   
**`2in.`**   
**`14n3an`**   
**`4inar`**   
**`Incel4`**   
**`in3car`**   
**`41nd`**   
**`lnSdling 21na`** 

**`13nee`**   
**`Iner4ar ISneaa`** 

**`4inga`**   
**`4inga`**   
**`inBgen`**   
**`41ngl`**   
**`lnSgllng 4ingo`** 

`J` 

**`4ingu`**   
**`2inl`**   
**`lfinl.`**   
**`14nia`**   
**`in3io`**   
**`inlis`**   
**`iEnlta. Einltio in3ity`** 

**`4 Ink`**   
**`41nl`**   
**`21nn`**   
**`2Uno`**   
**`14no4c`**   
**`ino4«`**   
**`14not`**   
**`2ina *`**   
**`' In3aa insurEa 2int.`**   
**`2in4tft inlu`**   
**`ISnus`**   
**`4iny`**   
**`21o`**   
**`41o.`**   
**`Iogo4`**   
**`io2gr`**   
**`ilol`**   
**`Io4a`**   
**`Ion3at ion4ory ion3i`** 

**`ioEph`**   
**`Ior31`**   
**`14o«`**   
**`ioEth`**   
**`lEoti`**   
**`Io4to`**   
**`14our`**   
**`2ip`**   
**`Ipa4`**   
**`iphra§4 Ip31`** 

**`Ip41c`**   
**`Ip4re4 Ip3ul`** 

**`13qua`**   
**`iqSuef`**   
**`Iq3uld`**   
**`Iq3ul3t 41r`** 

**`lira`**   
**`Ira4b`**   
**`14rac`**   
**`ird5e`**   
**`lretde 14rel`**   
**`14rel4 14res`** 

**`irEgl`**   
**`lrli`**   
**`Irl5do Ir4if`**   
**`iri3tu`**   
**`615r2iz`** 

**`ir4mln`**   
**`iro4g`**   
**`6iron. lrSnl`**   
**`21a.`**   
**`It5ag`**   
**`i*3ar`**   
**`ifa«S`**   
**`21tlc`**   
**`Is3ch`**   
**`41>a`**   
**`l<3er`**   
**`Sisl`**   
**`laShan i*3hon ish5op is31b`**   
**`Isl4d`**   
**`iEsls`**   
**`lsSltiT 41s4k`** 

**`islan4`**   
**`4iccs`**   
**`12co`**   
**`Iso5mor lalp`** 

**`Is2pi`**   
**`is4py`**   
**`4isls`**   
**`ii4sal is<on4 Ii4sea l<4ta. iilte`** 

**`liltl`**   
**`iet41y 41*tral 12«u`**   
**`lsSua`**   
**`4ita.`**   
**`Ita4bl 14tag`**   
**`41ta5m 13tan`**   
**`13tat`**   
**`21ta`**   
**`it3era ISteri`** 

**`it4ea`**   
**`21th`**   
**`ilti`**   
**`4itia`**   
**`412tle It31ca E15tlck It31g`**   
**`it5111 12tim`** 

**`21tlo`**   
**`41tla`**   
**`14tlsm`**   
**`12t5o5n 41 ton`** 

**`14traa itSry`** 

**`41tt`** 

**`•J`** 

**`It3nat`**   
**`lEtud`**   
**`It3ul`**   
**`4itz.`**   
**`iln`**   
**`2iT`**   
**`iT3ell`**   
**`iv3en.`**   
**`14v3er. 14ver«. lrSil.`** 

**`ii5io`**   
**`lTlit`**   
**`ISrora`**   
**`iv3o3ro 14v3ot`** 

**`415w`**   
**`Ix4o`**   
**`4iy`**   
**`41zar`**   
**`iz!4`**   
**`Slzont`**   
**`Sja`**   
**`jac4q`**   
**`ja4p`**   
**`lje`**   
**`jerSs`**   
**`4je«tie 4Je«ty`** 

**`jew3`**   
**`Jo4p`**   
**`6jndg`**   
**`3ka.`**   
**`k3ab`**   
**`kSag`**   
**`kala4`**   
**`kal4`**   
**`klb`**   
**`k2ed`**   
**`lkoa`**   
**`ke4g`**   
**`keSU`**   
**`k3en4d`**   
**`klar`**   
**`kea4`**   
**`k3eat.`**   
**`ke4ty`**   
**`k3f`**   
**`kh4`**   
**`kll`**   
**`Ekl.`**   
**`Sk21e`**   
**`k4111`**   
**`kiloS`**   
**`k4ia`**   
**`k41n.`**   
**`kin4da`**   
**`kSlneaa kln4g`** 

**`kl4p`**   
**`ki«4`**   
**`k51«h`**   
**`kk4`**   
**`kll`**   
**`4kley`**   
**`4kly`**   
`J` 

**`kla`**   
**`kEnea`**   
**`Ik2no`**   
**`koSr`**   
**`koah4`**   
**`k3on`**   
**`kroSn`**   
**`4kla2`**   
**`k4ae`**   
**`ka41`**   
**`May`**   
**`kSt`**   
**`klw`**   
**`Iab3ic`**   
**`14abo`**   
**`lacU`**   
**`Hade`**   
**`Ia3dy`**   
**`Iag4n`**   
**`Iara3o`**   
**`31and`**   
**`Iau4dl`**   
**`lanSet`**   
**`Ian4ta`**   
**`Iar4g`**   
**`Iar3i`**   
**`Ias4a`**   
**`Ia6tan`**   
**`41atall 41atlr`**   
**`41»T`**   
**`Ia4r4a`**   
**`211b`**   
**`Ibin4`**   
**`411c2`**   
**`Ice4`**   
**`13cl`**   
**`21d`**   
**`12da`**   
**`Id4era`**   
**`Id4eri`**   
**`Idl4`**   
**`ldSla`**   
**`13dr`**   
**`14dri`**   
**`l«2a`**   
**`Ie4bl`**   
**`laftS`**   
**`Slag.`**   
**`Elagg`**   
**`Ia4nat`**   
**`leaSatlc 41an.`** 

**`3 lane`**   
**`filana. llant`** 

**`Ie3ph`**   
**`Ie4pr`**   
**`leraSb`**   
**`Ier4a`**   
**`31erg`**   
**`314eri`**   
**`14ero`**   
**`lo«2`**   
**`laSaca`**   
**`Blaaq`**   
`y` 

**`31eaa`**   
**`Eleaa.`**   
**`13ara`**   
**`Ier4er.`**   
**`leT4ara`**   
**`Iev4era`**   
**`31ay`**   
**`41eya`**   
**`211`**   
**`lEtr`**   
**`411g4`**   
**`lSga`**   
**`lgarS`**   
**`14gaa`**   
**`Igo3`**   
**`213h`**   
**`114ag`**   
**`U2aa`**   
**`liarElt`**   
**`114»«`**   
**`114ato`**   
**`HSbl`**   
**`Sllclo`**   
**`114cor`**   
**`411ca`**   
**`411ct.`**   
**`141cn`**   
**`13icy`**   
**`131da`**   
**`lldSar`**   
**`Slldl`**   
**`Ilt3ar`**   
**`14111`**   
**`11411`**   
**`filigata SUgh`** 

**`114gra`**   
**`311k`**   
**`414141`**   
**`•Iln4bl`**   
**`Ila31`**   
**`U4ao`**   
**`141a4p`**   
**`141na`**   
**`1141na`**   
**`Iln3aa`**   
**`Un31`**   
**`llnkSar`**   
**`USog`**   
**`414iq`**   
**`Ili4p`**   
**`lilt`**   
**`12it.`**   
**`Slltlca lSlStlct Ilr3ar`**   
**`Ills`**   
**`41j`**   
**`Ika3`**   
**`13kal`**   
**`Ika4t`**   
**`111`**   
**`141av`**   
**`121a`**   
**`ISlea`**   
**`131ac`**   
`y` 

**`131eg`**   
**`131al`**   
**`131a4n`**   
**`131a4t`**   
**`1121`**   
**`1211n4`**   
**`lSllna`**   
**`114e`**   
**`lloqalS`**   
**`llEoat`**   
**`161ov`**   
**`21a`**   
**`lEaat`**   
**`Ia31ng`**   
**`14aod`**   
**`I>on4`**   
**`211B 2`**   
**`31o.`**   
**`lobSal`**   
**`Io4cl`**   
**`4 lot`**   
**`Slogle`**   
**`ISogo`**   
**`Slogn`**   
**`Ioa3ar`**   
**`Elong`**   
**`Ion41`**   
**`13o3nii`**   
**`loodE`**   
**`Plop*.`**   
**`lopSl`**   
**`13opa`**   
**`Iora4`**   
**`Io4rato`**   
**`loSrla`**   
**`lorfios`**   
**`Eloa. < loaSat`** 

**`Slotophli Eloiophf Ioa4t`** 

**`Io4ta`**   
**`lonnSd`**   
**`21ont`**   
**`41or`**   
**`21p`**   
**`lpaSb`**   
**`13pha`**   
**`lfiphl`**   
**`lpSlng`**   
**`13plt`**   
**`14pl`**   
**`lEpr`**   
**`411r`**   
**`211a2`**   
**`14ac`**   
**`12aa`**   
**`14.1a`**   
**`41t`**   
**`Itiag`**   
**`Itana5`**   
**`llta`**   
**`lUa4`**   
**`lt«ra«`**   
**`1U31`**   
**`Utlaa.`**   
`y`  
**78 T\\jX82 HYPHENATION PATTERNS** 

**`Hif4`**   
**`lltr`**   
**`Itu2`**   
**`Itur3i`**   
**`luBa`**   
**`In3br`**   
**`Iuch4`**   
**`Iu3ci`**   
**`Iu3en`**   
**`laM`**   
**`luBld`**   
**`In4aa`**   
**`Eluai`**   
**`lSumn.`**   
**`61aanla Iu3o`**   
**`Iuo3r`**   
**`41up`**   
**`Iuaa4`**   
**`Im3t«`**   
**`lint`**   
**`lEren`**   
**`15ret4`**   
**`211*`**   
**`117`**   
**`41ya`**   
**`41yb`**   
**`lyBae`**   
**`lySno`**   
**`21ya4`**   
**`lfyaa`**   
**`laa`**   
**`2mab`**   
**`•a2ca`**   
**`BaBchlM aa4cl`** 

**`EagSia`**   
**`Saagn`**   
**`2mah`**   
**`naidS`**   
**`4mald`**   
**`na31ig`**   
**`BtaSlln •al41i`**   
**`nal4ty`**   
**`Baanla`**   
**`canBli`**   
**`man31t`**   
**`4map`**   
**`Biaorine. aaEriz`**   
**`mar41y`**   
**`•ar3v`**   
**`maSac* nas4e`** 

**`aaalt`**   
**`5mate`**   
**`nath3`**   
**`na3tia`**   
**`4matiia 4mlb`** 

**`Bba4t5`**   
**`mSbll`**   
**`!>4b3ing nbl4r`** 

**`4»5c`** 

**`./`**   
**`4 M .`**   
**`2ned`**   
**`4 med.`**   
**`Soedla aeSdie •SeBdy •e2g`**   
**`aelEon me!4t`** 

**`ne2«`**   
**`aemloS lmen`**   
**`aen4a`**   
**`menBac nen4d« 4 mono`** 

**`ncn41`**   
**`msni4`**   
**`men«u6 3ment`**   
**`aen4t« neBon`**   
**`aSeraa 2nes`** 

**`Smesti ae4ta`**   
**`•et3al oelte`**   
**`meSthi a4atr`**   
**`Snetric BeStri* Eie3try Be4T`**   
**`4mlf`**   
**`2ah`**   
**`6al.`**   
**`nl3a`**   
**`•id4a`**   
**`•Id4g`**   
**`rig4`**   
**`Smllit •SiSlis •4111`**   
**`•In4a`**   
**`SBlnd`**   
**`aSinee a4ingl BlnSgli aSingly ain4t`**   
**`a4inu`**   
**`BlOt4`**   
**`a2ia`**   
**`nia4er. BlaBl`** 

**`aia4tl B51atry 4alth`** 

**`u21i`**   
**`4mk`**   
**`4all`**   
**`ill`**   
**`BuiaSry 4aln`** 

**`«jv4a,/`**   
**`J`** 

**`a4nin`**   
**`nn4o`**   
**`IBO`**   
**`4mocr`**   
**`Snocratiz mo2dl`** 

**`Bo4gO`**   
**`Bola2`**   
**`aoiSaa`**   
**`4BOIC`**   
**`aoSleat`**   
**`uo3me`**   
**`BonSet`**   
**`Bon6g«`**   
**`aonl3a`**   
**`Bon4iia`**   
**`aon4iat`**   
**`ao3niz`**   
**`aonol4`**   
**`ao3ny.`**   
**`ao2r`**   
**`4aora.`**   
**`aoa2`**   
**`moSsey . ao3sp`**   
**`aoth3`**   
**`nSouf`**   
**`3mou*`**   
**`BO2 T`**   
**`4mlp`**   
**`nparaS`**   
**`npaSrab`**   
**`aparSl`**   
**`a3pet`**   
**`aphaa4`**   
**`a2pl`**   
**`mpi4a`**   
**`apSiet`**   
**`o4plin`**   
**`nSpir`**   
**`mpBif`**   
**`npo3ri`**   
**`apoaSlt* a4poua`**   
**`•poTS`**   
**`ap4tr`**   
**`m2py`**   
**`4n3r`**   
**`4mla2`**   
**`B4ah`**   
**`mSal`**   
**`4at`**   
**`laa`**   
**`nalaSr4`**   
**`5 mult`**   
**`BUlti3`**   
**`3moa`**   
**`mun2`**   
**`4 sup`**   
**`au4n •`**   
**`4a*`**   
**`laa`**   
**`2nla2b`**   
**`n4abu '`**   
**`4aac.`**   
**`na4ca`**   
`J` 

**`nSact`**   
**`nagSer. nak4`** 

**`na41i`**   
**`naBlla`**   
**`4nalt`**   
**`naSmit`**   
**`n2an`**   
**`nancl4`**   
**`nan4it`**   
**`nank4`**   
**`nar3c`**   
**`4nare`**   
**`nar31`**   
**`nar41`**   
**`nBara`**   
**`n4aa`**   
**`na«4c`**   
**`nasBti`**   
**`n2at`**   
**`na3tal`**   
**`natoSmiz n2au`** 

**`nau3se`**   
**`3naut`**   
**`nav4o`**   
**`4nlb4`**   
**`ncarS`**   
**`n4ces.`**   
**`n3cha`**   
**`nScheo`**   
**`nSchil`**   
**`n3chis`**   
**`nclin`**   
**`nc4it`**   
**`ncourBa nlcr`**   
**`nlcu`**   
**`n4dal`**   
**`nSdan`**   
**`nlde`**   
**`ndSoat. ndl4b`** 

**`n5d2if`**   
**`nldit`**   
**`r.3dii`**   
**`nSduc`**   
**`ndu4r`**   
**`nd2we`**   
**`2ne.`**   
**`n3ear`**   
**`ne2b`**   
**`nob3u`**   
**`no2c`**   
**`Bneck`**   
**`2ned`**   
**`ne4gat`**   
**`negSatlr Bnega`** 

**`ne41a`**   
**`nelSlz`**   
**`ne5ai`**   
**`ne4mo`**   
**`lnon`**   
**`4nene`**   
**`3neo`**   
`J` 

**`ne4po`**   
**`ne2q`**   
**`nler`**   
**`neraSb`**   
**`n4erar`**   
**`n2ere`**   
**`n4erBi`**   
**`ner4r`**   
**`lnei`**   
**`2ne«.`**   
**`4neip`**   
**`2neat`**   
**`4nea*`**   
**`Snetlc`**   
**`ne4r`**   
**`nSera`**   
**`ne4*`**   
**`n3f`**   
**`n4gab`**   
**`n3gel`**   
**`nge4n4e n&gere`**   
**`nSgeri`**   
**`ngSha`**   
**`n3gib`**   
**`nglin`**   
**`n5git`**   
**`n4gla`**   
**`ngov4`**   
**`ng&th`**   
**`nigu`**   
**`n4gum`**   
**`n2gy`**   
**`4nlh4`**   
**`nhs4`**   
**`nhab3`**   
**`nhe4`**   
**`3n4ia`**   
**`ni3ar>`**   
**`ni4ap`**   
**`ni3la`**   
**`ni4bl`**   
**`ni4d`**   
**`niSdi`**   
**`ni4er`**   
**`nl2fl`**   
**`niEficat nBigr`** 

**`nik4`**   
**`nils`**   
**`nl3mii`**   
**`niin`**   
**`Bnine.`**   
**`nin4g`**   
**`ni4o`**   
**`Snli.`**   
**`nis4ta`**   
**`n2it`**   
**`r.4ith`**   
**`3nitio`**   
**`n3itor`**   
**`nl3tr`**   
**`nlj`**   
**`4nk2`**   
**`n5kero`**   
**`n3ket`**   
`J` 

**`nk3J.n`**   
**`nlkl`**   
**`4nll`**   
**`nEm`**   
**`nne4`**   
**`iunet4`**   
**`4nln2`**   
**`nne4`**   
**`nni3al`**   
**`nni4T`**   
**`nob41`**   
**`no3bl«`**   
**`nSocl`**   
**`4n3o2d`**   
**`3noe`**   
**`4nog`**   
**`noge4`**   
**`noieBi`**   
**`no514i`**   
**`Bnologia 3nomic`**   
**`nBoSmia`**   
**`no4mo`**   
**`no3roy`**   
**`no4n`**   
**`non4ag`**   
**`nonSi`**   
**`n5oniz`**   
**`4nop`**   
**`Enop5oSli norSab`** 

**`no4rary`**   
**`4nosc`**   
**`nos4e`**   
**`noaSt`**   
**`noSta`**   
**`lnou`**   
**`3noun`**   
**`nov3ol3`**   
**`no*13`**   
**`nlp4`**   
**`npl4`**   
**`npre4c`**   
**`nlq`**   
**`nlr`**   
**`nru4`**   
**`2nl«2`**   
**`nsBab`**   
**`nsatl4`**   
**`ns4c`**   
**`n2se`**   
**`n4e3ea`**   
**`naldl`**   
**`nslg4`**   
**`n2«l`**   
**`na3n`**   
**`n4soc`**   
**`ns4pe`**   
**`nfispi`**   
**`nsta5bl`**   
**`nit`**   
**`nta4b`**   
**`nter3»`**   
**`nt21`**   
**`n5tib`**   
**`nti4or`**   
`J` 

**`nti2f`**   
**`n3tin«`**   
**`n4t31ng nti4p`** 

**`ntrolSli nt4»`**   
**`ntu3ae`**   
**`nula`**   
**`nu4d`**   
**`nu5en`**   
**`nuMfe`**   
**`n3uln`**   
**`3nu31t`**   
**`n4ua`**   
**`nulma`**   
**`nBuml`**   
**`3nu4n`**   
**`n3uo`**   
**`nu3tr`**   
**`nlr2`**   
**`nl«4`**   
**`nyn4`**   
**`nyp4`**   
**`4nz`**   
**`n3za`**   
**`4oa`**   
**`oad3`**   
**`oSaSlea oard3`** 

**`oa84e`**   
**`oastBa`**   
**`oatBi`**   
**`ob3a3b`**   
**`oSbar`**   
**`obe41`**   
**`olbl`**   
**`o2bin`**   
**`obSing`**   
**`o3br`**   
**`ob3ul`**   
**`oSco`**   
**`och4`**   
**`o3chet`**   
**`oclf3`**   
**`o4cll`**   
**`o4clam`**   
**`o4cod`**   
**`oc3rac`**   
**`ocfiratli ocre3`** 

**`Bocrit`**   
**`octorBa oc3ula`** 

**`oScure`**   
**`odBded`**   
**`od31c`**   
**`odi3o`**   
**`o2do4`**   
**`odor3`**   
**`odSuct. od5ucta o4el`** 

**`oBeng`**   
**`o3er`**   
**`oe4ta`**   
**`ooeY`** 

**`o2fl`**   
**`oi51te oflt4t o2gSaCr ogBatlr o4gato`**   
**`olge`**   
**`oSgen* oSgeo`** 

**`o4ger`**   
**`o3gie`**   
**`lolgia`**   
**`og31t`**   
**`o4gl`**   
**`o6g21y 3ognlz o4gro`**   
**`oguBi`**   
**`logy`**   
**`2ogyn`**   
**`olh2`**   
**`ohab5`**   
**`012`**   
**`olc3ea oi3der oiff4`** 

**`oig4`**   
**`oiBlet o3ing`**   
**`ointSer oSiaa`**   
**`olSaon olatBen oi3ter`**   
**`oB]`**   
**`2ok`**   
**`o3ken`**   
**`ok51«`**   
**`olla`**   
**`o41an`**   
**`olam4`**   
**`ol2d`**   
**`oldie`**   
**`ol3ar`**   
**`o31eac o31et`**   
**`ol4fi`**   
**`ol21`**   
**`o311a`**   
**`o311ce olBid. o3114f`** 

**`oBlil`**   
**`ol31ng`**   
**`06II0`**   
**`oBlia. ol31sh`** 

**`oSlite oSlltlo oBliT`**   
**`011146`**   
**`olSoglz olo4r`**   
**`olSpl`**   
**`ol2t`**   
**`ol3ub`**   
**`J`** 

**`ol3ua«`**   
**`olSun`**   
**`oElua`**   
**`O12 T`**   
**`o21y`**   
**`onSah`**   
**`oraaBl`**   
**`pmSatlt oa2ba`** 

**`on4bl`**   
**`O2B «`**   
**`oa3ena`**   
**`0Bi5eraa o4aet`** 

**`oaBatry o3ola`** 

**`o&3ic.`**   
**`oa31ca`**   
**`oBald`**   
**`omlln`**   
**`oSalni`**   
**`Boounend 0B04g*`** 

**`o4aon`**   
**`oa3pi`**   
**`ocproS`**   
**`o2n`**   
**`onla`**   
**`on4ac`**   
**`o3nan`**   
**`onlc`**   
**`3oncll`**   
**`2ond`**   
**`onBdo`**   
**`o3nen`**   
**`onEeat`**   
**`on4gu`**   
**`onllc`**   
**`o3nlo`**   
**`cnllf`**   
**`o6nlu`**   
**`on3koy`**   
**`on4odl`**   
**`on3oay`**   
**`on3i`**   
**`onapl4`**   
**`or,«plrBa ontu4`** 

**`onten4`**   
**`on3t41`**   
**`ontlfS`**   
**`onBua`**   
**`onraB`**   
**`oo2`**   
**`oodBe`**   
**`oodSl`**   
**`oo4k`**   
**`oop31`**   
**`o3ord`**   
**`ooatE`**   
**`o2pa`**   
**`opeSd`**   
**`opler`**   
**`3opera`**   
**`4operag 2oph`** 

**`./`**  
**TgXBJ HYPHENATION PATTERNS 79** 

**`eSpham oSphar epSing o3pit`**   
**`oEpon`**   
**`o4poil olpr`**   
**`oplu`**   
**`opy5`**   
**`olq`**   
**`olra`**   
**`oSra.`**   
**`o4r3ag orEaliX orSanga or«5»`** 

**`oSraal orSal`** 

**`oraSsh orSeat. oraw4`** 

**`or4gu`**   
**`4oSrla or3<C4 o5ril`**   
**`orlln`**   
**`elrlo`**   
**`or3ity o3riu`** 

**`or2«i`**   
**`orn2a`**   
**`oSrof`**   
**`orSoug orfipa`** 

**`Sorrh`**   
**`or4aa`**   
**`or*Sen orst4`** 

**`oi3thi or3thy or4ty`** 

**`oErua`**   
**`olry`**   
**`oa3al`**   
**`oa2c`**   
**`o«4ca`**   
**`o3scop 4oscopl oSscr`**   
**`o.4i4e OSSitiT o«3ito oa3ity osl4u`**   
**`o«41`**   
**`o2so`**   
**`os4pa`**   
**`os4po`**   
**`oi2ti`**   
**`o5statl os5til o«5tit o4tan`**   
**`otale4g ot3er. otSara J`** 

**`o4Ua`**   
**`4oth`**   
**`othSeal oth314`**   
**`otSlc. ctSica`**   
**`o3tlca`**   
**`oStif`**   
**`o3tls`**   
**`otoSa`**   
**`ou2`**   
**`ou3bl`**   
**`oachSl ouSet`** 

**`ou41`**   
**`ouncEar oun2d`**   
**`OUE T`**   
**`or4en`**   
**`orer4na «-. arSa`** 

**`or4art o3vi§`** 

**`OTltl4 o5v4ol`** 

**`ow3dar ow3al`** 

**`owSaat owli`** 

**`ownSi`**   
**`o4»o`**   
**`oyla`**   
**`lpa`**   
**`pa4ca`**   
**`pa4ca`**   
**`pac4t`**   
**`p4ad`**   
**`Spagan`**   
**`p3agat p4ai`** 

**`pain4`**   
**`p4al`**   
**`pan4a`**   
**`panSal pan4ty pa3ny`**   
**`palp`**   
**`pa4pu`**   
**`paraSbl parSaga parSdl 3para`** 

**`parSal p4a4rl par41a`**   
**`paCta`**   
**`paStar`**   
**`Spathic paSthy`** 

**`pa4tric pav4`** 

**`Spay`**   
**`4plb`**   
**`pd4`**   
**`4pe.`**   
**`3pe4a`** 

**`J`** 

**`pear41 po2c`** 

**`2p2ed`**   
**`Speda`**   
**`Spedl`**   
**`padla4 ped4ic p4aa`**   
**`pee4d`**   
**`pak4`**   
**`pe41a`**   
**`peli4a pa4nan p4enc`** 

**`pen4th paSon`** 

**`p4ara. paraSbl p4erag p4ari`** 

**`parlfiat per4mal parneS p4arn`**   
**`parSo`**   
**`par3ti peEru`** 

**`par I T`**   
**`pe2t`**   
**`paStan paStlx 4pf`**   
**`4pg`**   
**`4ph.`**   
**`pharSi phe3no ph4er`** 

**`ph4ea. phlic`**   
**`Sphie`**   
**`phSing Ephisti 3phiz`** 

**`ph21`**   
**`3phob`**   
**`Sphona Ephonl pho4r`** 

**`4phs`**   
**`ph3t`**   
**`Epliu`**   
**`lphy`**   
**`pi3a`**   
**`plan4`**   
**`pi4cia pl4cy`** 

**`p41d`**   
**`pSlda`**   
**`pi3de`**   
**`Spidl`**   
**`3plec`**   
**`pi3en`**   
**`pl4grap pl31o`** 

**`pl2n`**   
**`p41n.`**   
**`' J`** 

**`plnd4`**   
**`p4ino`**   
**`3p j lo`**   
**`pion4`**   
**`p3ith`**   
**`piStha`**   
**`pi2ta`**   
**`2p3k2`**   
**`Ip212`**   
**`Splan`**   
**`plaaSt`**   
**`pll3a`**   
**`pllEer`**   
**`4plig`**   
**`pll4n`**   
**`plol4`**   
**`plu4«`**   
**`plnn4b`**   
**`<pl«`**   
**`2p3n`**   
**`po4c`**   
**`Spod.`**   
**`poSaa`**   
**`poSetS`**   
**`Epo4g`**   
**`poin2`**   
**`Epolnt`**   
**`polySt`**   
**`po4ni`**   
**`po4p`**   
**`Ip4or`**   
**`po4ry`**   
**`lpos`**   
**`pools`**   
**`p4ot`**   
**`po4ta`**   
**`Bpoun`**   
**`4plp`**   
**`ppa5ra`**   
**`p2pe`**   
**`piped`**   
**`p5pel`**   
**`p3pen`**   
**`p3per`**   
**`p3pet`**   
**`ppoSslta pr2`** 

**`pray4a`**   
**`Spreci`**   
**`preSco`**   
**`pre3em`**   
**`prefSac pre41a`** 

**`pre3r`**   
**`p3resa`**   
**`3presa`**   
**`preStan pre3v`** 

**`5prl4a`**   
**`prin4t3 pri4a`** 

**`prls3o`**   
**`p3roca`**   
**`profElt pro31`**   
**`pros3a`**   
`J` 

**`prolt`**   
**`2pls2`**   
**`p2aa`**   
**`ps4h`**   
**`p4aib`**   
**`2plt`**   
**`pt5»4b p2t«`**   
**`p2th`**   
**`ptl3aj`**   
**`ptu4r`**   
**`p4tv`**   
**`pub 3`**   
**`puo4`**   
**`puf4`**   
**`pulSc`**   
**`po4a`**   
**`pu2n`**   
**`pur4r`**   
**`Epus`**   
**`pu2t`**   
**`Eputa`**   
**`putSsr pu3tr`** 

**`put4tad put4tln P3w`**   
**`qu2`**   
**`quaSr`**   
**`2qua.`**   
**`3quer`**   
**`3quat`**   
**`2rab`**   
**`ra3bl`**   
**`rach4e r5acl`** 

**`rai5fi raf4t`** 

**`r2al`**   
**`ra41o`**   
**`ran3et r2ami`** 

**`rane5o ran4ga r4ani`** 

**`raSno`**   
**`rap3er 3raphy rarSc`**   
**`rare4`**   
**`rarSef 4raril r2aa`** 

**`ratlon4 rau4t`** 

**`raSvai ray3el raSilo rib`**   
**`r4bab`**   
**`14 bag`**   
**`rbi2`**   
**`rbi4f`**   
**`r2bin`**   
**`rSblna rbSlng.`** `J` 

**`rb4o`**   
**`rlc`**   
**`r2ca`**   
**`rcan4 r>e r3cha*`**   
**`rch4ar`**   
**`r4cl4b`**   
**`rc4it`**   
**`rcu>3`**   
**`r4dal`**   
**`rd21`**   
**`rdl4a`**   
**`rdl4er`**   
**`rdln4`**   
**`rdSlng`**   
**`2ra.`**   
**`ralal`**   
**`re3an`**   
**`raSarr`**   
**`Sraav`**   
**`re4i*`**   
**`rEabrat raeSoll recEonpa re4cre`** 

**`2r2ad`**   
**`ralda`**   
**`re3dis`**   
**`radSit`**   
**`re4fac`**   
**`ra2fa`**   
**`reSfar. re3H`** 

**`ra4fy`**   
**`reg3is`**   
**`re5it`**   
**`rolli`**   
**`reSlu`**   
**`r4en4ta ren4ta`** 

**`relo`**   
**`roSpln`**   
**`re4posi relpn`**   
**`rler4`**   
**`r4erl`**   
**`rero4`**   
**`reSrn`**   
**`r4ea.`**   
**`re4spl`**   
**`resaSib res2t`** 

**`reSstal reSstr`**   
**`re4tar`**   
**`re4t!4x re3tri`** 

**`reu2`**   
**`reSutl`**   
**`rey2`**   
**`re4Tal`**   
**`rar3al`**   
**`rSeySer. reSvors`** 

**`reSyerC reS?ll`**   
`J`   
**`rer5ol« r»4wh`** 

**`rlf`** 

**`r4fy`**   
**`rg2`**   
**`rg3ar`**   
**`r3gat`**   
**`rSgie`**   
**`rgl4n`**   
**`rgSlng`**   
**`rEgls`**   
**`rEglt`**   
**`rlgl`**   
**`rgo4n`**   
**`r3g«`**   
**`rh4`**   
**`4rh.`**   
**`4rhil`**   
**`rl3a`**   
**`ri*4b`**   
**`rl4ag`**   
**`r4ib`**   
**`ribSa`**   
**`ricEas`**   
**`r4ica`**   
**`4ricl`**   
**`Srlcld`**   
**`ri4cla`**   
**`r41co`**   
**`ridSar`**   
**`ri3anc`**   
**`riSent`**   
**`riler`**   
**`rlSet`**   
**`rlgEaft`**   
**`6rigi`**   
**`ril3ix`**   
**`Erinan`**   
**`rimSi`**   
**`3rimo`**   
**`rU4pa`**   
**`r2ina`**   
**`Srina.`**   
**`rin4d`**   
**`rin4o`**   
**`rin4g`**   
**`rilo`**   
**`Srlph`**   
**`rlphSa`**   
**`ri2pl`**   
**`ripSlie r4iq`** 

**`r31i`**   
**`r4ia.`**   
**`ri«4c`**   
**`r3ish`**   
**`rla4p`**   
**`ri3ta3b rSltad. ritEer. ritSers rit31c`**   
**`ri2tu`**   
**`rltSur`**   
**`rivSel`**   
**`J`**   
**`ritSat`**   
**`rirSi`**   
**`r3j`**   
**`r3kat`**   
**`rk41a`**   
**`rk41in`**   
**`rll`**   
**`rl«4`**   
**`r21ad`**   
**`r4Ug`**   
**`r41is`**   
**`rlElak`**   
**`rSlo4`**   
**`rla`**   
**`r«a5c`**   
**`r2aa`**   
**`rSaaa`**   
**`raEera`**   
**`raSlng`**   
**`r4Blng. r4alo`** 

**`rSait`** 

**`r4«y`**   
**`r4nar`**   
**`r3n*l`**   
**`r4ner`**   
**`rSnat`**   
**`rSnay`**   
**`rSnlc`**   
**`rlnls4`**   
**`rSnit`**   
**`rSnlT`**   
**`rno4`**   
**`r4aoa`**   
**`r3n«`**   
**`robSl`**   
**`r2oc`**   
**`roScr • ro4a`**   
**`rolia`**   
**`roSfll`**   
**`rok2`**   
**`roEker`**   
**`Erole.`**   
**`roa>Sat« rom4i`** 

**`rom4p`**   
**`ron4al`**   
**`ron4e`**   
**`roSn4ia ron4ta`** 

**`lrooa`**   
**`Sroot`**   
**`ro3pel`**   
**`rop3ic`**   
**`ror3i`**   
**`roSro`**   
**`ros5par ros4a`** 

**`ro4tha`**   
**`ro4ty • ro4va`** 

**`rovSol`**   
**`rox5`**   
**`rip`**   
**`r4paa`**   
**`/`**   
**`r!pr«t rpSar. r3pat`** 

**`rp4h4`**   
**`rpSing r3po`** 

**`rlr4`**   
**`rre4e`** 

**`rr«4l`**   
**`r4reo`**   
**`rre4ft`** 

**`rrl4a`**   
**`rrt4»`**   
**`rrou4`**   
**`rroa4`**   
**`rry§4`**   
**`4ra2`**   
**`rlaa`**   
**`naBtl`** 

**`r*4c`**   
**`r2ta`**   
**`rSaac`**   
**`raa4cr raEar. r»3aa`**   
**`raaSrS rlak`**   
**`rSaha`**   
**`rial`**   
**`r4sl4b raonS`** 

**`rlap`**   
**`rEav`**   
**`rtach4 r4tag`** 

**`rStab`**   
**`rt«n4d`**   
**`rtaSo`**   
**`rltl`**   
**`rtBlb`**   
**`rtl4d`**   
**`r4tior`**   
**`r3tlg`**   
**`rtllSt rtiUl`**   
**`r4tily r4tlat r4tlT`** 

**`r3tri`**   
**`rtroph4 rt4ah`**   
**`ru3a`**   
**`ru3e41 ru3en`**   
**`ru4gl`**   
**`ru3ia`**   
**`nw3pl`**   
**`ru2a`**   
**`runkS`**   
**`run4ty`**   
**`rSaac`**   
**`rutlEn`**   
**`rv4e`**   
**`rvel41`**   
**`r3ven`**   
**`ryBer.`** 

`v/`  
**80 1£X8\> HYPHENATION PATTERNS** 

**`rEreet rSrey`** 

**`r3ric`**   
**`TT14 T`**   
**`r3ro`**   
**`rlw`**   
**`>7<c`**   
**`Srynga rjr3t`** 

**`•a2`**   
**`2* lab`**   
**`Stack`**   
**`tacSrl tSaet`** 

**`Stai`**   
**`•»lir4 •a!4a`** 

**`taSlo`**   
**`•al4t`**   
**`3aanc`**   
**`•an4da • lap`**   
**`•aSta`**   
**`SiaStle •atSn`**   
**`•au4`**   
**`•aSTor fiia*`** 

**`4«Sb`**   
**`•can4tS • ca4p`** 

**`•cavS`**   
**`• 4ced`**   
**`4scei`**   
**`e4cee`**   
**`•ch2`**   
**`•4che`**   
**`3s4cla Eacin4d ecleS`** 

**`•tell`**   
**`•coM`**   
**`4icopy •conr5a alcn`**   
**`4eSd`**   
**`4te.`**   
**`•e4a`**   
**`seai4`**   
**`aeaSw`**   
**`•e2c3o 3eect`** 

**`4i4ed`**   
**`•e4d4a • Sedl`** 

**`•»2g`**   
**`•eg3r`**   
**`5aei`**   
**`•ella`**   
**`heelt`**   
**`Eselr`**   
**`A s erne`**   
**`•e4aol lenSat 4senc`** 

**`•en4d`** 

**`/`** 

**`eSened »»i.5g`** 

**`•Benin 4»entd 4tentl •ep3a3 4il«r. • 4erl`**   
**`«er4o`**   
**`4fervo tla4«`** 

**`•eSah`**   
**`•eiSt`**   
**`EieSua Slav`**   
**`»ey3en •aw4i`** 

**`6iez`**   
**`4«3f`**   
**`2«3g`**   
**`• 2h`**   
**`2§h.`**   
**`• Mar Ether`** 

**`thlin`**   
**`•h3io`**   
**`3ahip`**   
**`ahlvS`**   
**`«ho4`**   
**`•hSuld • ho.i3 •hor4`**   
**`•hortS 4ihw`** 

**`silb`**   
**`•Slcc`**   
**`Slide. Eaidee Etidi`** 

**`•iEdii 4«igna • il4e`**   
**`4til7 2»lin`** 

**`•2ina`**   
**`Seine. •3ing`**   
**`leio`**   
**`Salon`**   
**`•lonSa • 12r`**   
**`•irSa`**   
**`leie`**   
**`3altlo Eiiu`** 

**`lliT`** 

**`Saiz`**   
**`•kS`**   
**`4ske`**   
**`•3ket`**   
**`•kEine akEing • 112`** 

**`•31at`**   
**`•21e`**   
**`•llthS J`** 

**`2»1«`**   
**`•3ma`**   
**`anal13 •aanS`**   
**`aael4`**   
**`eSnen`**   
**`Eaalth ••olEd4 •In4`** 

**`leo`**   
**`•o4ce`**   
**`•oft3`**   
**`•o41ab •ol3d2 •o311c EeolY`** 

**`S»o«`**   
**`3a4on. iona4`**   
**`•on4g`**   
**`•4op`**   
**`Saophle •Sophls •Sophy •orEc`**   
**`•orSd`**   
**`4*OT`**   
**`•oSrl`**   
**`2ipa`**   
**`Sepal`**   
**`•pa4n`**   
**`•pen4d 2«5peo 2iper`**   
**`•2phe`**   
**`Sapher •phoS`** 

**`•pil4`**   
**`ipSing 4«pio`** 

**`.4ply`**   
**`•4pon`**   
**`•por4`**   
**`4spot`**   
**`•qual41 air`**   
**`2ee`**   
**`•lea`**   
**`••aa3`**   
**`•2eSc`**   
**`•3eel`**   
**`•5aeng e4iei. •S*et`** 

**`• lei`**   
**`»4«le`**   
**`••14er •s5ily • 4al`** 

**`••411`**   
**`•4sn`**   
**`••pend4 ••2t`**   
**`••urSa ••Ew`** 

**`2a t.`**   
`y` 

**`>2tag`**   
**`•2tal`**   
**`•tan4i`**   
**`Estand`**   
**`•4ta4p`**   
**`Satat.`**   
**`•4ted`**   
**`iternSl •Staro`** 

**`•ta2v`**   
**`•tewSa`**   
**`•3tha`**   
**`•t21`**   
**`•4tl.`**   
**`•Stla`**   
**`title`**   
**`Sstick •4tla`**   
**`•3tif`**   
**`•tSlng`**   
**`Estlr`**   
**`• ltla`**   
**`Sstock`**   
**`•tom3a`**   
**`£•tone •4 top`**   
**`Sitore`**   
**`•t4r`**   
**`•4trad`**   
**`Estratu «4tray`** 

**`•4trid`**   
**`4atry`**   
**`4et3w`**   
**`»2ty`**   
**`In`**   
**`•ulal`**   
**`su4b3`**   
**`«u2g3`**   
**`8U5il`**   
**`suit3`**   
**`«4ul`**   
**`•u2o`**   
**`•un3i`**   
**`EU2JI`**   
**`•u2r`**   
**`4«T`**   
**`•w2`**   
**`4iwo`**   
**`•4y`**   
**`4»yc`**   
**`Ssyl`**   
**`•yn5o`**   
**`•y5ri*`**   
**`lta`**   
**`3ta.`**   
**`2tab`**   
**`taSblsa Stabollt 4taci`**   
**`ta5do`**   
**`4taM`**   
**`taiSlo`**   
**`ta21`**   
**`taSla`**   
**`talSen`**   
**`J`** 

**`talSi`**   
**`4talk`**   
**`tal41i« taSlog taSmo`** 

**`tan4da tanta3 taSper taSpl`** 

**`tar4a`**   
**`4 tare`**   
**`4 tare`**   
**`ta3rlx ta*4e`** 

**`taSiy`**   
**`4tatie ta4tur taun4`** 

**`tav4`**   
**`2taw`**   
**`taz41e 2tlb`**   
**`lie`**   
**`t4ch`**   
**`tchSet 4tld`** 

**`4ta.`**   
**`t.ead41 4te»t`**   
**`teco4`**   
**`Etect`**   
**`2tled`**   
**`teSdi`**   
**`ltee`**   
**`t«g4`**   
**`te5ger te5gi`**   
**`3tel.`**   
**`teli4`**   
**`Btele`**   
**`ta2ma2 teo3at 3tenan 3 tone`**   
**`3 tend`**   
**`4tenee ltont`** 

**`ten4tag lteo`**   
**`te4p`**   
**`te5pa`**   
**`t«r3c`**   
**`Ster3d lteri`**   
**`ter5ie» ter3is terlEza Sternit terSv`**   
**`4tos.`**   
**`4te«a`**   
**`t3os». tethSa 3teu`** 

**`3 tax`**   
**`4tey`** 

`•J` 

**`2tlt`**   
**`4tlg`**   
**`2th.`**   
**`than4`**   
**`th2e`**   
**`4 the*`**   
**`th3ei»`**   
**`theSat`**   
**`tho31e`** 

**`Sthat`**   
**`thSle. thBica`** 

**`4thll`**   
**`Sthink`**   
**`4thl`** 

**`thEoda Ethodlc 4thoo`** 

**`thorSlt thoSris 2th«`**   
**`ltla`**   
**`tl«ab`**   
**`U4ato`**   
**`2tl2b`**   
**`4tick`**   
**`t4ico`**   
**`t4iclu`**   
**`Etidl`**   
**`3tlan`**   
**`tlf2`**   
**`tlBfy`**   
**`2tig`**   
**`Stign`**   
**`tillSin ltiB`**   
**`4timp`**   
**`tim5ul`**   
**`2tlin`**   
**`t2ina`**   
**`3tine.`**   
**`3tini`**   
**`ltio`**   
**`ti5oc`**   
**`tionSaa 5tiq`** 

**`ti3>a`**   
**`3ti«e`**   
**`tia4n`**   
**`tl5eo`**   
**`tie4p`**   
**`Sti»tica ti3tl`** 

**`ti4a`**   
**`ltiT`**   
**`tiv4a`**   
**`ltiz`**   
**`tl3za`**   
**`ti3zon`**   
**`2tl`**   
**`tSla`**   
**`tlan4`**   
**`3tlo.`**   
**`3tled`**   
**`3tle».`**   
**`tSlet.`**   
`y` 

**`tSle`**   
**`4tl«`**   
**`t*o4`**   
**`2tln2`**   
**`lto`**   
**`to3b`**   
**`toScrat 4todo`**   
**`2tof`**   
**`to2gr`**   
**`toSlc`**   
**`to2aa`**   
**`tom4b`**   
**`to Say`**   
**`ton4all toSnat`**   
**`4tono`**   
**`4 tony`**   
**`to2ra`**   
**`to3rie`** 

**`torBls`**   
**`toi2`**   
**`Etour`**   
**`4 tout`**   
**`toSvar`**   
**`4tlp`**   
**`ltrt`**   
**`traSb`**   
**`traSck`**   
**`tracl4`**   
**`trac41t trac4ta tra»4`**   
**`tra5ven travSelS treSf`**   
**`tr«4a`**   
**`tremSl`**   
**`Etria`**   
**`triSce* Etricla 4trica`** 

**`2tria`**   
**`tri4T`**   
**`troSai`**   
**`tronSi`**   
**`4trony`**   
**`tro5pha tro3ep`**   
**`tro3r`**   
**`truSi`**   
**`true4`**   
**`4tle2`**   
**`t4ec`**   
**`tsh4`**   
**`t4cw`**   
**`4t3t2`**   
**`t4tei`**   
**`tSto`**   
**`ttu4`**   
**`ltu`**   
**`tula`**   
**`tu3ar`**   
**`tu4bl`**   
**`tud2`**   
**`4tue`**   
**`4tnf4`**   
**`EtttSi`** 

**`Stoa`**   
**`U4nlt`**   
**`2tSup.`**   
**`Stura`**   
**`Sturl`**   
**`tnrSla`**   
**`turBo`**   
**`tuSry`**   
**`3tu«`**   
**`4tT`**   
**`t»4`**   
**`4tlva`**   
**`t«i«4`**   
**`4tTO`**   
**`lty`**   
**`4tya`**   
**`2tyl`**   
**`typaS`**   
**`tySph`**   
**`4ti`**   
**`tx4e`**   
**`4mb`**   
**`u»c4`**   
**`uaEnt`**   
**`nan4t`**   
**`uarSant uar2d`** 

**`narSl`**   
**`uarSt`**   
**`ulat`**   
**`uav4`**   
**`ub4a`**   
**`u4bel`**   
**`u3ber`**   
**`u4bara`**   
**`ulb41`**   
**`u4bSlhg • u3bla. u3ca`** 

**`ucl4b`**   
**`uc4it`**   
**`ncla3`**   
**`u3cr`**   
**`u3cu`**   
**`u4cy`**   
**`udSd`**   
**`ud3er`**   
**`udEest`**   
**`udev4`**   
**`oldie`**   
**`ud3ied`**   
**`uJ3ie«`**   
**`ud5i«`**   
**`uSdit`**   
**`u4don`**   
**`ud4sl`**   
**`u4du`**   
**`u4ena`**   
**`uens4`**   
**`uen4ta`**   
**`uer411`**   
**`3ufa`**   
**`u3fl`**   
**`ugh3en`**   
**`J`** 

**`mgSU`**   
**`2nl2`**   
**`nilBli`** 

**`«14n`**   
**`ullng`**   
**`ulr4a`**   
**`ulta4`**   
**`U1T3`**   
**`«lT4ar. afj`**   
**`4ok`**   
**`nllt`**   
**`ul*Bb`**   
**`nBliti`**   
**`alch4`**   
**`Eulcha`**   
**`ulSder`**   
**`«14a`**   
**`«Uaa`**   
**`nl4gl`**   
**`ul2i`**   
**`nSlla`**   
**`ulSlng`**   
**`ulSlsh`**   
**`vl41ar`**   
**`ul4114b nl411f`**   
**`4ulSa`**   
**`ull4o`**   
**`4ul.`**   
**`ul«5aa`**   
**`nlltl`**   
**`«ltra9`**   
**`4«ltn`**   
**`a31n`**   
**`ulBul`**   
**`ulSr`**   
**`uaEab`**   
**`\im4bi`** 

**`ua4bly`**   
**`ulal`**   
**`n4a31ng oaorSo`**   
**`ua2p`**   
**`unat4`** 

**`n2na`**   
**`un4er`**   
**`ulnl`**   
**`un4ia`**   
**`u2nin`**   
**`nnSiah`**   
**`IU\13T`**   
**`un3»4`**   
**`un4ow`**   
**`unt3ab`**   
**`un4ter. un4tee`** 

**`unu4`**   
**`unEy`**   
**`unSz`**   
**`u4ore`**   
**`nSo*`**   
**`ulou`**   
**`ulpe`**   
**`uperS*`** 

**`uSpla`**  
**\\ njX82 HYPHENATION PATTERNS81** 

**`opSlng`**   
**`. a3pl`**   
**`np3p`**   
**`apportS`**   
**`nptSlb`**   
**`aptu4`**   
**`olra`**   
**`4ura.`**   
**`n4rag`**   
**`u4ra» , or4ba`** 

**`nrc4`**   
**`arid`**   
**`uraEat`**   
**`ur4far`**   
**`ur4fr`**   
**`u3rif •`**   
**`urHflc`**   
**`urlin`**   
**`uSrlo`**   
**`nlrit`**   
**`urSis`**   
**`or2i`**   
**`arising.`**   
**`ur4no`**   
**`uro»4`**   
**`ur4pe`**   
**`nr4pl`**   
**`artSar`**   
**`orStaa`**   
**`ur3the`**   
**`urti4`**   
**`uMtla`**   
**`o3ro`**   
**`2ua`**   
**`uEsad`**   
**`oSaan .`**   
**`aa4ap`**   
**`o»c2`**   
**`uaScl`**   
**`uaeSa`**   
**`uSala`**   
**`u3aic`**   
**`ui411n`**   
**`uilp`**   
**`uaSil`**   
**`utStar*`**   
**`ualtr`**   
**`n2«u`**   
**`osur4`**   
**`uta4b`**   
**`u3tat`**   
**`4uta.`**   
**`4utel`**   
**`4uten`**   
**`uten4i`**   
**`4ult21`**   
**`at!Sill`**   
**`u3tina`**   
**`ut3ing`**   
**`utlonSa`**   
**`u4tia`**   
**`' SuEtlx`**   
**`u4tll`**   
**`utSof`**   
**`utoSg`** 

**`ntoGnatlc oSton`**   
**`u4tou`**   
**`Qtf4`**   
**`o3u`**   
**`uu4«`**   
**`Q1T2`**   
**`uru3`**   
**`nx4a`**   
**`. Ira`**   
**`Era.`**   
**`2rla4b`**   
**`racSll`**   
**`rac3n`**   
**`rag4`**   
**`Ta4g«`**   
**`T.Elia`**   
**`TalEo`**   
**`T»llU`**   
**`raSao`**   
**`TaEnix`**   
**`raSpl`**   
**`varSlad`**   
**`Srat`**   
**`4T6.`**   
**`4rad`**   
**`TegS`**   
**`T3«1.`**   
**`T«13H`**   
**`ra41o`**   
**`T4ely`**   
**`Ten3oa`**   
**`TEanua`**   
**`T4erd`**   
**`ETara.`**   
**`T4eral`**   
**`T3eran`**   
**`verSenc`**   
**`v4eraa`**   
**`rer3ie`**   
**`Terni4n`**   
**`3reraa`**   
**`Tor3th`**   
**`T4e2«`**   
**`4rea.`**   
**`Tea4ta`**   
**`ve4ta`**   
**`tet3or`**   
**`re4ty`**   
**`TlSall`**   
**`fiTlan`**   
**`Erida.`**   
**`- Srlded`**   
**`4r3iden`**   
**`Erldaa`**   
**`STldi`**   
**`T311`**   
**`yiSgn`**   
**`vik4`**   
**`2Til`**   
**`Evllit`**   
**`v3i31ix`**   
**`Tlin`**   
**`4ri4na`**   
**`T2inc`**   
**`TlnSd`**   
**`J`** 

**`4Tlng`**   
**`rio31`**   
**`T3io4r`**   
**`TllOU`**   
**`Ti4p`**   
**`TlSro`**   
**`Til3it`**   
**`T13IO`**   
**`Tl3au`**   
**`4Titi`**   
**`rit3r`**   
**`4Tlty`**   
**`3T1T`**   
**`E TC`**   
**`T0i4 . STO)C`**   
**`To41a`**   
**`TSola`**   
**`ETOlt`**   
**`STOIT`**   
**`TOBSI`**   
**`TorEab`**   
**`Tori 4`**   
**`To4ry . To4ta`** 

**`4T0ta«`**   
**`4TT4`**   
**`T4y`**   
**`wSabl`**   
**`2wac`**   
**`waSgar`**   
**`wagSo`**   
**`waits`**   
**`wSal.`**   
**`waa4`**   
**`war4t`**   
**`wat4t`**   
**`walta`**   
**`waSTer`**   
**`wlb`**   
**`weaSria reath3`** 

**`wed4n`**   
**`weet3`**   
**`weeST`**   
**`wel41`**   
**`wlar`**   
**`weatS`**   
**`w3eT`**   
**`whi4`**   
**`vlS`**   
**`wil2`**   
**`wlllSin wln4de`** 

**`wln4g`**   
**`wir4`**   
**`3wisa`**   
**`«ith3`**   
**`wixS`**   
**`w4k`**   
**`il4o«`**   
**`wl31n`**   
**`w4no`**   
**`Iwo2`**   
**`woal`**   
**`soSren`**   
**`J`** 

**`»5p`**   
**`Tra4`**   
**`wri4`**   
**`wrlta4`**   
**`w3ih`**   
**`wt41`**   
**`wi4pa`**   
**`wEi4t`**   
**`4wt`** 

**`zla`**   
**`xacSa`**   
**`z4ago`**   
**`zao3`**   
**`z4ap`**   
**`zaaS`**   
**`z3c2`**   
**`zla`**   
**`ie4coto z2ed`**   
**`zer41`**   
**`zeSro`** 

**`x)h`**   
**`xhl2`**   
**`zhilS`**   
**`zhu4`**   
**`z3i`**   
**`zlSa`**   
**`ziEc`**   
**`ziSdi`**   
**`x41ma`**   
**`xiEmlz`**   
**`z3o`**   
**`z4ob`**   
**`x3p`**   
**`xpan4d`**   
**`xpectoS xpe3d`**   
**`zlt2`**   
**`z3tl`**   
**`Zla`**   
**`xu3a`**   
**`zx4`**   
**`ySae`**   
**`3yar4`**   
**`y5at`**   
**`ylb`**   
**`yi<=`**   
**`y2ca`**   
**`ycSer`**   
**`y3ch`**   
**`ycMa`**   
**`ycoa4`**   
**`ycoU`**   
**`yld`**   
**`yEea`**   
**`ylar`**   
**`y4arf`**   
**`ye»4`**   
**`ye4t`** 

**`ysgi`**   
**`4y3h`** 

**`yii`**   
**`y31a`**   
**`ylla5bl y31o`** 

**`J`**   
**`y51tt`**   
**`yabolS y»e4`**   
**`ynpaS`**   
**`yn3chr ynSd`**   
**`ynfig`**   
**`ynBlc`**   
**`Bynx`**   
**`ylo4`**   
**`yoEd`**   
**`y4oSg`**   
**`yoa4`**   
**`yoEnat y4ona`** 

**`y4ot`**   
**`y4ped`**   
**`yperB`**   
**`yp3i`**   
**`y3po`**   
**`y4poc`**   
**`yp2U`**   
**`yBptt`**   
**`yraEa`**   
**`yrSla`**   
**`y3ro`**   
**`yr4r`**   
**`ya4c`**   
**`y3«2«`**   
**`yt31ca yi3io`** 

**`3yai(`**   
**`y4io`**   
**`y»f4`**   
**`yslt`**   
**`y»3ta`**   
**`ygar*. y3thi« yt3ic`** 

**`ylw`**   
**`zal`**   
**`z5a2b`**   
**`zar2`**   
**`4zb`**   
**`2x*`**   
**`ze4n`**   
**`za4p`**   
**`zler`**   
**`ze3ro`**   
**`xat4`**   
**`2x11`**   
**`i41* •1 *`**   
**`z41a`**   
**`Exl`**   
**`4za`**   
**`lzo`**   
**`zo4a`**   
**`zoBol`**   
**`zta4`**   
**`4zlz2`** 

**`x4zy`** 

**`<`**  
`Answers` 

**`moun-tain-ous vil-lain-ous be-tray-al de-fray-al por-tray-al hear-ken`** 

**`ex-treme-ly su-preme-ly`** 

**`tooth-aches`** 

**`bach-e-lor ech-e-lon`** 

**`.riff-raff`** 

**`anal-o-gous ho-mol-o-gous`** 

**`gen-u-ine`** 

**`any-place`** 

**`co-a-lesce`** 

**`fore-warn fore-word`** 

**`de-spair`** 

**`ant-arc-tic corn-starch`** 

**`mast-odon`** 

**`squirmed`** 

**`82`**  
References 

\[1\] Knuth, Donald E. *T\&Land* METflFONT, *New Directions in Typesetting.* Digital Press, 1979\. 

\[2\] *Webster's Third New International Dictionary.* **G.** & **C. Merriam, 1961\.** \[3\] Knuth, Donald E. *The WEB System of Structured Documentation.* Preprint, Stanford Computer Science Dept., September 1982\. 

\[4\] Knuth, Donald E. Tie *Art of Computer Programming, Vol. 3, Sorting **and** Searching.* Addison-Wesley, 1973\. 

\[5\] Standish, T. A. *Data Structure Techniques.* Addison-Wesley, 1980\. 

\[6\] Aho, A. V., Hopcroft, J. E., and Ullman, J. D. *Algorithms and Data Structures.* Addison-Wesley, 1982\. 

\[7\] Bloom, B. Space/time tradeoffs in hash coding with allowable **errors.** *CACM 13,* July 1970, 422-436. 

\[8\] Carter, L., Floyd, R., Gill, J., Markowslcy, G., and Wegman, M. Exact and approximate membership testers. *Proc. 10th ACM SIGACT Symp.,* 1978, 59- 65\. 

\[9\] de la Briandais, Rene. File searching using variable length keys. *Proc. Western Joint Computer Conf. x5,* 1959, 295-298. 

\[10\] Fredkin, Edward. Trie memory. *CACM 3,* Sept. 1960, 490-500. \[11\] TVabb Pardo, Luis. Set representation and set intersection. Ph.D. thesis, Stan ford Computer Science Dept., December 1978\. 

\[12\] Mehlhorn, Kurt. Dynamic binary search. *SIAM J. Computing 8,* May 1979, 175-198. 

\[13\] Maly, Kurt. Compressed tries. *CACM 19,* July 1976, 409-415. \[14\] Knuth, Donald E. *Tj$L82.* Preprint, Stanford Computer Science Dept., Septem ber 1982\. 

\[15\] Resnikoff, H. L. and Dolby, J. L. The nature of affixing in written English. *Mechanical Translation 8,* 1965, 84-89. Part II, June 1966, 23-33. \[16\] Tie *Merriam-Webster Pocket Dictionary.* G. & C. Merriam, 1974\. \[17\] Gorin, Ralph. SPELL.REG\[UP,DOC\] at SU-AI. 

\[18\] Peterson, James L. Computer programs for detecting and correcting spelling errors. *CACM 23,* Dec. 1980\. 673-687. 

**83**  
84 REFERENCES 

**\[19\] Nix, Robert. Experience with a space-efficient way to store a dictionary. *CACM 24,* May 1981, 297-298.** 

**\[20\] Morris, Robert and Cherry, Lorinda L. Computer detection of typographical errors. *IEEE Trans. Prof. Comm. PC-18,* March 1975, 54-64.** 

**\[21\] Downey, P., Sethi, R., and Tarjan, R. Variations on the common subexpression problem. *JACM 27,* Oct. 1980, 758-771.** 

**\[22\] Tarjan, R. E. and Yao, A. Storing a sparse table. *CACM 22,* Nov. 1979,608-611.** 

**\[23\] Zeigler, S. F. Smaller faster table driven parser. Unpublished manuscript, Madi son Academic Computing Center, U. of Wisconsin, 1977\.** 

**\[24\] Aho, Alfred V. and Ullman, Jeffrey D. *Principles of Compiler Design,* sections 3.8 and 6.8. Addison-Wesley, 1977\.** 

**\[25\] Pfleeger, Charles P. State reduction in incompletely specified finite-state ma chines. *IEEE Trans. Computers C-22,* Dec. 1973, 1099-1102.** 

**\[26\] Kohavi, Zvi. *Switching and Finite Automata Theory,* section 10-4. McGraw Hill, 1970\.** 

**\[27\] Knuth, D. E., Morris, J. H., ar. i Pratt, V. R. Fast pattern matching in string\*. *SIAM J. Computing 6,* June 1977, 323-350.** 

**\[28\] Aho, A. V. In R. V. Book (ed.), *Formal Language Theory: Perspectives and Open Problems.* Academic Press, 1980\.** 

**\[29\] Kucera, Henry and Francis, W. Nelson. *Computational Analysis of Present-Day American English.* Brown University Press, 1967\.** 

**\[30\] Research and Engineering Council of the Graphic Arts Industry. Proceedings of the 13th Annual Conference, 1963\.** 

**\[31\] Stevens, M. E. and Little, J. L. Automatic *Typographic-Quality Typesetting Techniques: A State-of-the-Art Review.* National Bureau of Standards, 1967\.** 

**\[32\] Berg, N. Edward. *Electronic Composition, A Guide to the Revolution in Type setting.* Graphical Arts Technical Foundation, 1975\.** 

**\[33\] Rich, R. P. and Stone, A. G. Method for hyphenating at the end of a printed line. *CACM 8,* July 1965, 444-445.** 

**\[34\] Wagner, M. R. The search for a simple hyphenation scheme. Bell Laboratories Technical Memorandum MM-71-1371-8.** 

**\[35\] Gimpel, James F. *Algorithms in Snobol 4\.* Wiley-Interscience, 1976\.**  
**REFERENCES 85** 

\[36\] **Ocker,** Wolfgang A. A program to hyphenate English words. *IEEE **Trans. Prof.** Comm. PC-18,* June 1975, 78-84. 

\[37\] Moitra, A., Mudur, S. P., and Narwekar, A. W. Design and analysis of **a** hy phenation procedure. *Software Prac. Exper.* 0, 1979, 325-337. 

\[38\] Lindsay, R., Buchanan, B. G., Feigenbaum, E. A., and Lederberg, J. *DENDRAL.* McGraw-Hill, 1980\.