# environmental_audio_analysis
A machine learning model for recognizing multiple audio events
 To run program just run the 'app.py' as we run regularly, after that it will run the app on localhost at address 127.0.0.1/5000:
 
 For more details I have also added presentation and report of project.

Monitoring human and social activities is becoming increasingly pervasive in our living environments. Automated systems, that use multimodal techniques employing both video and audio information, have recently gained importance. When visual cues cannot reliably recognize the activities (events) and environments/contexts, audio cues are complementary to visual cues. The information collected from a semantic audio analysis can be beneficial and related applications such as analyzing and forecasting patterns of events, classifying/searching audio records, customer alerts, and robot navigation.Thus,Build an environmental audio scene and event recognition system which recognizes scenes and events solely on the basis of audio. And classify the given audio event and audio scene on the basis of its credibility towards children which helps to understand impact of the event on children.

Algorithms & Models :
Model:
Sequential Model
Algorithms:
Spectral gating algorithm
Optimization Algorithm   : Adam Optimization Algorithm
Activation Function : Relu,Softmax
Loss function : Mean squared error

Experiments and Results:

1st Generation -
We had built a 7 layer CNN model with Adam optimizer, Categorical crossentrophy as loss function and 100 epochs we got training accuracy of 65.43 % and validation accuracy of 15.03%.

2nd Generation -
In 2nd generation we built a CNN model with 5 layers with Adam optimizer, sparse categorical cross entropy as loss function and 200 epochs resulting in training accuracy of 64.12% and validation accuracy of 19.04%

3rd Generation -
In 3rd generation ,we built a 4 layer model having all 4 dense layers with mean squared error as loss function, adam optimizer,  and relu,softmax as activation functions giving training accuracy of 80.45% and validation accuracy of 25.3%

4th Generation -
For 4th generation we added a single dropout layer and a flattening layer to reduce the overfitting along with previous generations specs as it is resulting in a whopping  accuracy of 92.32% and validation accuracy of 34.68%

5th Generation -
In 5th generation we used categorical crossentrophy as loss function, it resulted in accuracy of more than 90% but it was also yielding higher validation loss as compared to other generations

6th Generation -
This is the latest generation we have tested so far and the best generation we have build. With similar specs as in 4th generation along with an extra dropout layer it resulted in training accuracy of 95.97% and first time we got validation accuracy more than 40% (specifically speaking 41.53%)
