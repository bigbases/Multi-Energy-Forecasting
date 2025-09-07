**Multi-Energy-Forecasting: Learning With Correlation-Guided Attention for Multienergy Consumption Forecasting**

This repository contains the official implementation for the paper "Learning With Correlation-Guided Attention for Multienergy Consumption Forecasting," published in *IEEE Transactions on Industrial Informatics*.

---

## 💡 Key Contributions

* **Correlation-Guided Attention Mechanism:** We introduce a novel attention module that uses a two-stage learning strategy to dynamically capture and reflect the time-varying correlations between different energy sources. This approach improves prediction performance by steering the model based on the relationships between energy types.
* **Two-Stage Learning Strategy:** Our model is trained in two distinct stages to optimize two types of losses: a correlation loss and a prediction loss. This progressive training method first aligns attention weights with actual correlations and then fine-tunes the model for accurate forecasts.
* **Performance Validation:** We validated our model's effectiveness through extensive experiments on real-world datasets, demonstrating consistent performance improvements over baseline models. Our method can be integrated with various existing time-series models (e.g., MLP, LSTM) as a feature extractor.

---

## 📈 Performance Highlights

Our model consistently outperforms conventional methods, achieving significant improvements in Mean Absolute Error (MAE) for various energy sources.

* **Electricity:** 14.27% improvement in MAE.
* **Water:** 10.07% improvement in MAE.
* **Gas:** 3.14% improvement in MAE.
* **Hot Water:** 9.95% improvement in MAE.

---

For more details on the model architecture, loss functions, and comprehensive results, please refer to the full research paper: [https://ieeexplore.ieee.org/document/10604923](https://ieeexplore.ieee.org/document/10604923).