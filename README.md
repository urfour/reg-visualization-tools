# reg-visualization

A Visualization for Comparative Analysis of Regression Models

## Installation

```bash
pip install -r requirements.txt
```

## How to use

### Plots

To generate the plots as shown in the article, please use the following command:

```bash
python main.py -ap
```

### Models

The functions defined use a DataFrame containing the predictions and the errors of the models, in the following format:

| target | target_model1 | error_model1 | target_model2 | error_model2 |
|--------|---------------|--------------|---------------|--------------|
| target1| value1        | value2       | ...           | ...          |
| target2| value1        | value2       | ...           | ...          |
| ...    | ...           | ...          | ...           | ...          |

For instance, for two models predicting the price of houses:

| price  | price_model1 | error_model1 | price_model2 | error_model2 |
|--------|--------------|--------------|--------------|--------------|
| 154000 | 155400       | 1400         | 132600       | -21400       |
| 98450  | 98420        | -30          | 109500       | 11050        |

The script train.py can be used to train them:

```bash
python train.py --data <dataset> --target <target>
```

## Datasets and architectures

Table **1** describes the datasets used in our experiments, and Table **2** describes the architectures used for training for each dataset. As Datasets B to D are synthetically generated, the process to create them and their implementation are described in the repository.

Equations (1), (2), and (3) define the custom loss functions used for C-MAPSS and AI4I2020 models.

---

### Table 1: Description of the datasets used

| **No** | **Dataset**   | **Description**                                      | **Target**   |
| ------ | ------------- | ---------------------------------------------------- | ------------ |
| A      | C-MAPSS       | Turbofan degradation                                 | RUL          |
| B      | Synthetic 1   | Moderate errors vs Extreme errors                    | Dummy Target |
| C      | Synthetic 2   | Under and over estimations                           | Dummy Target |
| D      | Synthetic 3   | Similar errors but on different individuals          | Target       |
| E      | Apartment     | Evolution of the price of apartments for rent in USA | Price        |
| F      | AI4I2020      | Evolution of operational state of machinery          | RUL          |

---

### Table 2: Description of the architectures used

| **No** | **Model**         | **Criteria**        | **Dataset** |
| ------ | ----------------- | ------------------- | ----------- |
| 1      | LSTM              | Squared Error       | C-MAPSS     |
| 2      | LSTM              | Absolute Error      | C-MAPSS     |
| 3      | LSTM              | LIN-SE 3            | C-MAPSS     |
| 4      | LSTM              | LIN-SE 4            | C-MAPSS     |
| 5      | LSTM              | LIN-SE 5            | C-MAPSS     |
| 6      | LSTM              | LIN-SE 6            | C-MAPSS     |
| 7      | LSTM              | LIN-LIN 0.01 - 1.0  | C-MAPSS     |
| 8      | LSTM              | LIN-LIN 0.05 - 1.0  | C-MAPSS     |
| 9      | LSTM              | LIN-LIN 0.2 - 1.0   | C-MAPSS     |
| 10     | LSTM              | LIN-LIN 0.3 - 1.0   | C-MAPSS     |
| 11     | LSTM              | QUAD-QUAD 0.01      | C-MAPSS     |
| 12     | LSTM              | QUAD-QUAD 0.03      | C-MAPSS     |
| 13     | Moderate errors   |                     | Synthetic 1 |
| 14     | Extreme errors    |                     | Synthetic 1 |
| 15     | Under-estimations |                     | Synthetic 2 |
| 16     | Over-estimations  |                     | Synthetic 2 |
| 17     | Similar errors 1  |                     | Synthetic 3 |
| 18     | Similar errors 2  |                     | Synthetic 3 |
| 19     | Decision Tree     | Squared Error       | Apartment   |
| 20     | XGBoost           | Squared Error       | Apartment   |
| 21     | LSTM              | QUAD-QUAD 0.2       | AI4I2020    |
| 22     | LSTM              | QUAD QUAD 0.8       | AI4I2020    |

---

### Custom Loss Functions

Let $\hat{y}_i$ be the predicted value, $y_i$ the actual value, and $r = \hat{y}_i - y_i$.

1. **LIN-SE**:

$$
\text{LIN-SE}(a) = \frac{1}{N} \sum_{i=1}^N
\begin{cases}
-a(\hat{y}_i - y_i), & \text{if } r < 0 \\
(\hat{y}_i - y_i)^2, & \text{otherwise}
\end{cases}
$$

2. **LIN-LIN**:

$$
\text{LIN-LIN}(a, b) = \frac{1}{N} \sum_{i=1}^N
\begin{cases}
-a(\hat{y}_i - y_i), & \text{if } r < 0 \\
b(\hat{y}_i - y_i), & \text{otherwise}
\end{cases}
$$

3. **QUAD-QUAD**:

$$
\text{QUAD-QUAD}(a) = \frac{1}{N} \sum_{i=1}^N
\begin{cases}
2a(\hat{y}_i - y_i)^2, & \text{if } r < 0 \\
2(-a + 1)(\hat{y}_i - y_i), & \text{otherwise}
\end{cases}
$$
