# chatbot_2nd
python funny chatbot project which is created by tensorflow2.0 keras 
此專案利用自製intent和DNN建立chatbot，利用斷詞後轉成onehot encoding然後接兩層NN預測intent裡的類別，
最後搭配random.choice()從該類別中任意選擇回答，若預測類別中最大機率並沒有超過0.8則回答不知道你再問甚麼!

**!!!這個project還沒有用到lstm等RNN技巧，主要是自建intent為主**

**file description:**
- intent.json: 建立主要對話及類別
- data.pickle: 利用nltk取出intent.json訊息，必存於data.pickle中，利於後續使用
- model.bin: DNN 模型
- main.py: 執行檔

**執行方式:** 
```
python main.py
```

**結果**
![image](https://github.com/kevinlin19/chatbot_2nd/blob/master/pic/%E6%93%B7%E5%8F%96.PNG)
