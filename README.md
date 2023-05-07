Download Link: https://assignmentchef.com/product/solved-machine-learning-hw3
<br>
<h1></h1>

<ol>

 <li><strong> Perceptron Lower Bound. </strong>Show that for any 0 <em>&lt; γ &lt; </em>1 there exists a number <em>d &gt; </em>0, vector <em>w<sup>? </sup></em>∈ R<em><sup>d </sup></em>and a sequence of examples (<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>m</sub>,y<sub>m</sub></em>) such that:</li>

</ol>

(c) Perceptron makes  mistakes on the sequence.

(Hint: Choose and let {<em>x<sub>i</sub></em>}<em><sub>i </sub></em>be the standard basis of R<em><sup>d</sup></em>)

<ol start="2">

 <li><strong> Halving Algorithm. </strong>Denote by A<em><sub>Hal </sub></em>the Halving algorithm you have seen in class. Let <em>d </em>≥ 6, X = {1<em>,…,d</em>} and let H = {<em>h<sub>i,j </sub></em>: 1 ≤ <em>i &lt; j </em>≤ <em>d</em>} where</li>

</ol>

<em> .</em>

Show that <em>M</em>(A<em><sub>Hal</sub>,</em>H) = 2.

(Definition of mistake bound <em>M</em>(A<em>,</em>H): Let H be a hypothesis class and A an online algorithm. Given any sequence <em>S </em>= (<em>x</em><sub>1</sub><em>,h<sup>?</sup></em>(<em>x</em><sub>1</sub>))<em>,…,</em>(<em>x<sub>m</sub>,h<sup>?</sup></em>(<em>x<sub>m</sub></em>)) where <em>m </em>is an integer and <em>h<sup>? </sup></em>∈ H, let <em>M</em><sub>A</sub>(<em>S</em>) be the number of mistakes A makes on the sequence <em>S</em>. Then <em>M</em>(A<em>,</em>H) = sup<em><sub>S </sub>M</em><sub>A</sub>(<em>S</em>)).

<ol start="3">

 <li><strong>(15 points) Interval growth function. </strong>The goal of this exercise is to compute the growth function of the interval hypothesis class <em>H </em>= {<em>h<sub>a,b </sub></em>: <em>a &lt; b</em>} where <em>h<sub>a,b</sub></em>(<em>x</em>) = 1 if <em>x </em>∈ [<em>a,b</em>] and 0 otherwise. In other words, your goal is to give an explicit expression to Π<em><sub>H</sub></em>(<em>m</em>) = max<em><sub>C</sub></em><sub>&#x1f610;<em>C</em>|=<em>m </em></sub>|<em>H<sub>C</sub></em>| where <em>H<sub>C </sub></em>is the restriction of <em>H </em>on the set <em>C</em>.</li>

 <li><strong>(15 points) Sample complexity of agnostic PAC. </strong>Let <em>H </em>be a hypothesis class of functions from a domain X to {0<em>,</em>1} and let the loss function be the 0-1 loss. Assume that <em>V Cdim</em>(<em>H</em>) = <em>d &lt; </em>∞. Show that if</li>

</ol>

then

Pr[

1

2                                                                                              <em>Handout Homework 3: April 5, 2020</em>

To prove the above claim you can use the following lemma without proving it:

Lemma: Let <em>a </em>≥ 1 and <em>b &gt; </em>0. Then: <em>x </em>≥ 4<em>a</em>log(2<em>a</em>) + 2<em>b </em>→ <em>x </em>≥ <em>a</em>log(<em>x</em>) + <em>b</em>.

You can also assume that <em>δ </em>is as small as you desire.

<ol start="5">

 <li><strong>(15 points) Prediction with Log-Loss. </strong>Consider a prediction setting with input <em>X </em>and true label <em>Y </em>∈ {0<em>,</em>1} (i.e., it has two possible values: zero and one). The predictor <em>h</em>(<em>x</em>;<em>θ</em>) returns a number in [0<em>,</em>1] via the function</li>

</ol>

<em>.                                                     </em>(1)

Here <em>θ</em>(<em>x</em>) is a function of <em>x </em>which defines <em>h</em>. Note that this construction makes sure that <em>h</em>(<em>x</em>) ∈ [0<em>,</em>1].

Consider the loss function (this is an instance of the well-known cross-entropy loss, which we will learn more about):

∆(<em>y,y</em>ˆ) = −<em>y </em>log(ˆ<em>y</em>) − (1 − <em>y</em>)log(1 − <em>y</em>ˆ)                                    (2)

Find the value of <em>θ</em>(<em>x</em>) that minimizes E[∆(<em>Y,h</em>(<em>X</em>;<em>θ</em>))]. Use these to express <em>h</em>(<em>x</em>) as a simple function of E[<em>Y </em>|<em>X</em>].

<em>Handout Homework 3: April 5, 2020                                                                                               </em>3

<h1>Programming Assignment</h1>

<u>Submission Guidelines:</u>

<ul>

 <li>Download the file skeleton perceptron.py from Moodle. In each of the following questions you should only implement the algorithm at each of the skeleton files. Plots, tables and any other artifact should be submitted with the theoretical section.</li>

 <li>In the file skeleton py there is an helper function. The function reads the examples labelled 0, 8 and returns them with 0-1 labels. If you are unable to read the MNIST data with the provided script, you can download the file from here: https://github.com/amplab/datasciencesp14/blob/master/lab7/mldata/mnist-original.mat.</li>

 <li>Your code should be written with Python 3.</li>

 <li>Make sure to comment out / remove any code which halts the code execution, such as matplotlib popup.</li>

 <li>Your submission should include exactly one file: py.</li>

</ul>

<ol>

 <li><strong>Perceptron. </strong>Implement the Perceptron algorithm (in the file name perceptron.py). Do not forget to normalize the samples to have unit length (i.e., k<em>x<sub>i</sub></em>k = 1).</li>

</ol>

<ul>

 <li><strong>(8 points) </strong>Use only the first <em>n </em>= 5<em>,</em>10<em>,</em>50<em>,</em>100<em>,</em>500<em>,</em>1000<em>,</em>5000 samples as an input to Perceptron. For each <em>n</em>, run Perceptron 100 times, each time with a different random order of inputs, and calculate the accuracy of the classifier on the test set of each run. You should therefore have 100 accuracy measurement per <em>n</em>. Print a table showing, for each <em>n</em>, the mean accuracy across the 100 runs, as well as the 5% and 95% percentiles of the accuracies obtained.</li>

 <li><strong>(4 points) </strong>The weight vector <em>w</em>, returned by Percepton, can be viewed as a matrix of weights, with which we multiply each respective pixel in the input image. Run Perceptron on the entire training set, and show <em>w</em>, as a 28×28 image, for example with imshow(reshape(image, (28, 28)), interpolation=’nearest’). Give an intuitive interpretation of this image.</li>

 <li><strong>(4 points) </strong>Calculate the accuracy of the classifier trained on the full training set, applied on the test set.</li>

 <li><strong>(4 points) </strong>Choose one (or two) of the samples in the test set that was misclassified by Perceptron (with the full training set) and show it as an image (show the unscaled images). Can you explain why it was misclassified?</li>

</ul>