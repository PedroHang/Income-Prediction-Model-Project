import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import patsy
from sklearn.metrics import r2_score

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Projeto 2 - Previsão de Renda",
     page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQPgOUkJM5NMRksqNtIelgm_b-cz29IVt_tfA&s",
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('./input/previsao_de_renda.csv')


renda_df = (renda
    .drop(columns=['Unnamed: 0', 'id_cliente', 'data_ref'])
    .dropna(subset=['tempo_emprego'])
    .drop_duplicates()
    .reset_index(drop=True)
)

renda_df = (renda_df
    .assign(tempo_emprego_idade_ratio = lambda x: x['tempo_emprego'] / x['idade'])
    .assign(log_renda = lambda x: np.log(x['renda']))
)

train_df, test_df = train_test_split(renda_df, test_size=0.2, random_state=40)

formula = (
    'log_renda ~ '
    'C(sexo) + '
    'C(posse_de_veiculo) * C(posse_de_imovel) + '
    'qtd_filhos + '
    'C(tipo_renda) + '
    'C(posse_de_imovel) + '
    'C(educacao) + '
    'C(estado_civil) + '
    'C(tipo_residencia) + '
    'C(tipo_residencia) + '
    'idade + '
    'tempo_emprego + '
    'qt_pessoas_residencia + '
    'tempo_emprego_idade_ratio'
)

y_train, X_train= patsy.dmatrices(formula_like=formula, data=train_df)
y_test, X_test = patsy.dmatrices(formula_like=formula, data=test_df)

## Ridge e Lasso

alphas = [0,0.001,0.005,0.01,0.05,0.1]
r2_ridge = []
r2_lasso = []

for alpha in alphas:
    
    modelo = sm.OLS(y_train, X_train)
    
    reg_ridge = modelo.fit_regularized(method='elastic_net',
                                     refit=True,
                                     L1_wt=0, #ridge
                                     alpha = alpha)
    
    reg_lasso = modelo.fit_regularized(method='elastic_net',
                                     refit=True,
                                     L1_wt=1, #Lasso
                                     alpha = alpha)
    
    y_pred_ridge = reg_ridge.predict(X_test)
    y_pred_lasso = reg_lasso.predict(X_test)
    aux = r2_score(y_test, y_pred_ridge)
    tmp = r2_score(y_test, y_pred_lasso)
    
    r2_ridge.append(aux)
    r2_lasso.append(tmp)

modelo = sm.OLS(y_train, X_train)
reg = modelo.fit_regularized(method='elastic_net',
                                     refit=True,
                                     L1_wt= 0, #ridge
                                     alpha = 0)

y_pred = reg.predict(X_test)
y_test = y_test.ravel()

predictions = pd.DataFrame({
    'Valor Verdadeiro (log)': y_test,
    'Valor Predito (log)': y_pred,
    'Diferença (Log)': (y_test - y_pred),
    'Valor Verdadeiro': np.exp(y_test), 
    'Valor Predito': np.exp(y_pred),
    'Diferença': ((np.exp(y_test)) - (np.exp(y_pred))),
})

