# reg-visualization-tools

Visualization Tools for Comparative Analysis of Regression Models

## Installation

```bash
pip install -r requirements.txt
```

## How to use
### ICTAI

To generate the plots as shown in the ICTAI article, please use the following command:
```bash
python ictai.py -train_all -plot
```
### Visualizations

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

