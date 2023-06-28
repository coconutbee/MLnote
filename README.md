中原大學鐵人賽
===

- [從0開始學習AI人工智慧演算法](/WaGAHPIaSE6fgpsVaZEjPA)


#### <span class="gray">自2000年左右，隨著大數據的興起，科學家們發現單純依靠大數據分析資料的局限性。雖然大數據能有效解決市場調查困難的問題，但因使用多數資料庫管理系統處理時效率低下；於是在2010年左右，科學家開始嘗試通過教導電腦從資料中學習，讓系統訓練演算法以尋找大型資料集中的模式和關聯，並根據這些分析做出最佳決策和預測；然而到了2015年左右，人們意識到機器學習的能力有限，無法像人類那樣思考和執行策略，因此人工智慧橫空出世，通過感知、學習、推理和校正等階段，使人工智慧能夠深入大量數據，執行複雜且繁瑣的任務，幫助人類突破限制，拓展研究和應用的範疇。</span>
#### 既然人工智慧這麼重要，那肯定是一個值得深入探討的專業啦:grin::grin: -----睡不飽的電機老屁股


## 第1章深度學習與機器學習
###  一、什麼是機器學習?
#### 機器學習(ML:Machine Learning)是人工智慧的一個分支，<span class="red">透過演算法將收集到的資料進行分類或預測模型訓練</span>，在未來中，當得到新的資料時，可以透過訓練出的模型進行預測 ，如果這些效能評估可以透過利用過往資料來提升的話，就叫機器學習。
* #### 機器學習的應用:天氣預測、人臉辨識、指紋辨識、車牌辨識、醫學診斷輔助、測謊、證卷分析、語音處理等。

<div style="text-align: center;">
<img src="https://www.fsm.ac.in/blog/wp-content/uploads/2022/08/ml-e1610553826718.jpg" width="2000" >
</div>

###  二、什麼是深度學習?
#### 深度學習是機器學習的分支領域，而且是以「層」的概念來建構演算法，進而創造出能夠自主學習並做出智慧決定的「人工神經網路」。它本質上是一個具有三層或更多層的神經網絡。這些神經網絡<span class="red">試圖模擬人腦的行為，使其能夠從大量數據中學習。</span>雖然具有單層的神經網絡仍然可以做出近似預測，但額外的隱藏層可以幫助優化和改進準確性。
* #### 深度學習的應用:自動駕駛、語音辨識、影像辨識、藝文圖片生成、自然語言處理等。
##### 值得一提的是，當今火紅的Chat gpt就是一種深度學習的產物，其背後對應的技術是自然語言處理。 
<div style="text-align: center;">
<img src="https://www.headmind.com/wp-content/uploads/2023/01/CHAT-GPT.png" width="400">

</div>

### 三、深度學習與機器學習的關係
#### 深度學習其實就是機器學習的一種，與機器學習不同的是，深度學習被設計來讓電腦可以處理大量的資料，並且簡化機器學習的過程。以下可以看出機器學習與深度學習的關係、過程。


<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/HkI3Hi3P2.png" width="300" >

##### 圖一、深度學習(綠)VS機器學習(黃)

<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*AXoNKxKw1_tGrT4aDCVjTQ.png" width="500">
    
##### 圖二、機器學習與深度學習的處理過程

</div>

###  四、機器學習的種類
#### 1.監督式學習(Supervised Learning):給予資料和其正確解答**來學習的方式
* #### 常見的監督式學習有:類神經網路、單純貝式分類器(Naive Bayes Classifier)、邏輯迴歸(Logistic Regression)、決策數(Decision Tree)、支援向量機(SVM:Supply Vector Machine)。
    #### 監督式學習的特色:能自動讓電腦<span class="red">高速且精準</span>地進行分類或判別工作。
#### 2. 非監督式學習(Unsupervised Learning):不給予正確答案、僅從資料來學習的方式
* #### 常見的非監督式學習有:K平均法(K-means)。
    #### 非監督式學習的特色:能使電腦更容易<span class="red">理解未知資料的分析手法。</span>
####  2.強化學習(RL:Reinforcement learning):強調如何基於環境而行動，以取得最大化的預期利益。 其靈感來源於心理學中的行為主義理論，即有機體如何在環境給予的獎勵或懲罰的刺激下，逐步形成對刺激的預期，產生能獲得最大利益的習慣性行為。
* ####    強化學習和標準的監督式學習之間的區別在於，<span class="red">它不需要出現正確的輸入/輸出，也不需要精確校正次優化的行為。</span>只需在探索（在未知的領域）和遵從（現有知識）之間找到平衡。
    #### 強化學習的特色:使電腦在沒有人類干預、沒有被寫入明確的執行任務程式下，就能夠做出一系列的決策，<span class="red">產生最佳結果的策略。</span>
    

| 名稱 | 監督式學習 | 非監督式學習| 強化學習|
| -------- | -------- | -------- | --------|
| 內容     |給電腦資料和正確答案學習| 只給電腦資料學習|不需要精確校正、優化 |
| 特色    |高速、精準判斷            |理解對未知資料的分析|不用人類指導，並完成最佳化計算|

維修中~~~
  ---
## 第2章深度學習-CNN卷積神經網路
### 一、什麼是CNN?
#### CNN卷積神經網路(CNN:Convolutional Neural Network)，是一種來自於模仿生物的視覺系統所開發出來的技術，透過模仿動物視覺皮層組織的神經連接方式，<span class="blue">單神經元</span>只對<span class="blue">有限區域</span>內的刺激做反應，不同神經元的感知區域相互重疊，從而覆蓋整個視野。
* #### 學習關鍵字:卷積操作、卷積核、感受野、特徵圖

<div style="text-align: center;">
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*irWQaiIjHS27ZAPaVDoj6w.png" width="800">
<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*RIBWK55dcDJa-zI_dFPDnw.png" width=1000>
    
##### CNN卷積概念圖
</div>

### 二、CNN的基礎知識
#### 以結構上來看，CNN可以分成三個部分:輸入層、隱藏層、輸出層。在輸入層的部分中存在著<span class="blue">卷積層(Convolution Layer)</span>，卷基層主要是透過特定數目的<span class="blue">卷積核(filter)</span>，對輸入的多通道<span class="blue">特徵圖</span>進行掃描和運算，進一步得到<span class="red">更高層語意資訊</span>的輸出特徵圖<span class="gray">(通道數目等於卷積核個數)</span>。

### 三、卷積運算
#### 
<style>
.red {
  color: red;
}
.gray {
  color: gray;
}

.blue {
  color: blue;
}
</style>