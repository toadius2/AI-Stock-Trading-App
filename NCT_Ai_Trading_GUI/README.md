AI-Crypto V2 

  

Process: 

 The program starts by executing the Robinhood algorithm and collecting data.  

 

The process takes a few minutes 

 

The GUI starts with the main screen 

 

The "aicryptogui.py" contains 4 main functions: 

   

  1- Arbitrage 

   

  2- Sentiment Analysis 

   

  3- Perfect Trade 

     

  4- Robinhood Algorithm 

  

   

  Arbitrage: 

   

    - Upon pressing "Start" button it executes the arbitrage script and displays outputs as labels. 

    - Press "Quit" to return to Main Screen 

     

  Sentiment Analysis: 

 

    - Clicking "Start" will start the analysis and scrap the web. (NOTE! News need to be updated as labels into the GUI. I am missing         something on the code that updates the labels) 

    - Press "Quit" to return to Main Screen 

 

Perfect Trade: 

 

    - Clicking “Start” will start fetching the latest data and updating the timestamp. If it is the first time running the Perfect Trade       this might take some time as it needs to train the Neural Network. 

 

      On the left of the terminal it can be noticed that there are 50 epochs to be trained.  

 

      After training into the 50 epochs the label should indicate a PEAK, NEUTRAL or VALLEY. Last time I implemented and tested the code       everything was working great. Tensorflow lately released a new version and is messing around with the epochs. The training does         not stop and goes on. As a result, the labels do not update. (NOTE! Tensorflow need to be updated or probably the code needs to be       changed accordingly.) 

    - Press "Quit" to return to Main Screen 

 

Robinhood Algorithm: 

 

    - Clicking “Start” will display the images from the data accumulation of the Robinhood algorithm. 

    - Press "Quit" to return to Main Screen 
