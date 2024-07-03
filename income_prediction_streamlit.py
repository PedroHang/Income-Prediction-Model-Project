import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Income Prediction Project",
     page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQPgOUkJM5NMRksqNtIelgm_b-cz29IVt_tfA&s",
     layout="wide",
)

st.markdown("""
    <style>
        .main-title {
            color: #FFD700;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        .sub-title {
            color: #ADFF2F;
            font-size: 2em;
            font-weight: bold;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }
        .section-title {
            color: #ADD8E6;
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }
        .section-content {
            font-size: 1.1em;
            line-height: 1.6;
        }
        .data-table {
            margin-top: 1em;
        }
        .data-table th, .data-table td {
            padding: 0.5em;
            border: 1px solid #ddd;
        }
        .data-table th {
            background-color: #333;
            color: white;
        }
    </style>
    
    <h2 class="main-title">Income Prediction Dashboard</h2>

    <h5 class="section-content">This dashboard showcases the process of exploratory data analysis on a dataset. The insights gained will be used to build a model for predicting an individual's income based on various factors.</h5>

    <hr>

    <h3 class="sub-title">Exploratory Analysis of the Data</h3>

    <hr>

    <h4 class="section-title">Data Dictionary</h4>

    <p class="section-content">
    Below is a complete data dictionary for our main dataset, including variable names, descriptions, and variable types.
    </p>

    <p class="section-content">
    It is also worth mentioning that the data presented below consists of an altered version of the original data for better utilization in our exploratory analysis.
    </p>

    <table class="data-table">
        <thead>
            <tr>
                <th>Variable</th>
                <th>Description</th>
                <th>Type</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>sexo</td>
                <td>Gender (Gender of the client)</td>
                <td>object</td>
            </tr>
            <tr>
                <td>posse_de_veiculo</td>
                <td>Vehicle ownership (Ownership of a vehicle or not)</td>
                <td>bool</td>
            </tr>
            <tr>
                <td>posse_de_imovel</td>
                <td>Property ownership (Ownership of property or not)</td>
                <td>bool</td>
            </tr>
            <tr>
                <td>qtd_filhos</td>
                <td>Number of children (The number of children the client has)</td>
                <td>int</td>
            </tr>
            <tr>
                <td>tipo_renda</td>
                <td>Income type (Entrepreneur, Employee, Public Servant, Pensioner, or Scholar)</td>
                <td>object</td>
            </tr>
            <tr>
                <td>educacao</td>
                <td>Education (Primary, Secondary, Incomplete Higher Education, Complete Higher Education, or Postgraduate)</td>
                <td>object</td>
            </tr>
            <tr>
                <td>estado_civil</td>
                <td>Marital status (Single, Married, Widowed, Union, or Separated)</td>
                <td>object</td>
            </tr>
            <tr>
                <td>tipo_residencia</td>
                <td>Type of residence (House, Governmental, With parents, Rent, Studio, or Community)</td>
                <td>object</td>
            </tr>
            <tr>
                <td>idade</td>
                <td>Age (Age of the client)</td>
                <td>int</td>
            </tr>
            <tr>
                <td>tempo_emprego</td>
                <td>Employment duration (in years) (Employment duration of the client)</td>
                <td>float</td>
            </tr>
            <tr>
                <td>qt_pessoas_residencia</td>
                <td>Number of residents (Number of people living in the client's residence)</td>
                <td>float</td>
            </tr>
            <tr>
                <td>renda</td>
                <td>Income (Income of the client) (Dependent variable)</td>
                <td>float</td>
            </tr>
            <tr>
                <td>tempo_emprego_idade_ratio</td>
                <td>Employment duration to age ratio (This is the division of employment duration by the individual's age. Values tend to be higher for those who have worked longer and are younger)</td>
                <td>float</td>
            </tr>
            <tr>
                <td>log_renda</td>
                <td>Logarithm of income (The natural logarithm of the individual's income. This metric helps in masking outliers)</td>
                <td>float</td>
            </tr>
        </tbody>
    </table>
""", unsafe_allow_html=True)


##################### Don't touch ####################################

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

############################ Dont't touch ################################

num_rows = st.slider('Select number of rows to display', min_value=5, max_value=100, value=9)

st.text("An example of how the dataset actually looks like:")
st.table(renda_df.head(num_rows))


scale_option = st.radio('Choose scale for income display:', ('Log Scale', 'Original Scale'))

bins = st.slider('Select number of bins to display', min_value=10, max_value=200, value=80)

if scale_option == 'Log Scale':
    # Histogram for 'log_renda'
    st.write("## Distribution of Income (Log Scale)")
    st.write("###### Note that, in this case, the values have been converted to the log scale so that outliers would not break the graph. (Some details might not make sense with this scale)")
    fig = px.histogram(renda_df, x='log_renda', nbins=bins, title='Income Distribution (Log Scale)', labels={'log_renda': 'Income'})
    # Convert log_renda back to original scale for x-axis labels
    tickvals = np.log([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    ticktext = ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000']
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    fig.update_traces(marker_line_width=1.5, marker_line_color="black")  # Customize bin separation
    st.plotly_chart(fig, use_container_width=True)

else:
    # Histogram for 'renda'
    st.write("## Distribution of Income (Original Scale)")
    fig = px.histogram(renda_df, x='renda', nbins=bins, title='Income Distribution', labels={'renda': 'Income'})
    fig.update_traces(marker_line_width=1.5, marker_line_color="black")  # Customize bin separation
    st.plotly_chart(fig, use_container_width=True)

# Plot Box Plots
if scale_option == 'Log Scale':
    st.subheader("Box Plot of Income (Log Scale)")
    st.write("This box plot shows the distribution of income on a logarithmic scale.")
    fig = px.box(renda_df, y='log_renda', title='Income Box Plot (Log Scale)', labels={'log_renda': 'Income'})
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)
else:
    st.subheader("Box Plot of Income (Original Scale)")
    st.write("This box plot shows the distribution of income on the original scale. It is almost impossible to get any insights from this type of visualization")
    fig = px.box(renda_df, y='renda', title='Income Box Plot', labels={'renda': 'Income'})
fig.update_layout(
    height=600,  # Increase the height to unsqueeze the box plot
    width=800,   # Increase the width to provide more space
    margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins to provide more padding around the plot
)
st.plotly_chart(fig, use_container_width=True)

if scale_option == 'Log Scale':
    st.subheader("Scatter Plot of Employment time vs. Income (Log Scale)")
    st.write("This scatter plot shows the relationship between employment duration and income on a logarithmic scale with a regression line.")
    fig = px.scatter(renda_df.head(2000), x='tempo_emprego', y='log_renda', trendline='ols', 
                     labels={'tempo_emprego': 'Employment time (years)', 'log_renda': 'Log of Renda'},
                     title='Income vs. Employment Time (years)',
                     height=600)  # Increase the height of the plot
    # Transform log_renda back to original scale for y-axis tick labels
    tickvals = np.log([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    ticktext = ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000']
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)
else:
    st.subheader("Scatter Plot of Employment time vs. Income (Original Scale)")
    st.write("This scatter plot shows the relationship between employment duration and income on the original scale with a regression line.")
    fig = px.scatter(renda_df.head(2000), x='tempo_emprego', y='renda', trendline='ols', 
                     labels={'tempo_emprego': 'Employment time (years)', 'renda': 'Income'},
                     title='Income vs. Employment Time (years)',
                     height=600)  # Increase the height of the plot

# Customize trendline color and scatter point size
fig.update_traces(marker=dict(size=5), line=dict(color='green'))

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)


label_encoders = {}
for column in renda_df.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    renda_df[column] = le.fit_transform(renda_df[column])
    label_encoders[column] = le

# Compute the correlation matrix
corr = renda_df.corr()

# Add description for the heatmap
st.write("""
## Correlation Heatmap

The heatmap below shows the correlation between different variables in the dataset. A correlation value closer to 1 indicates a strong positive relationship, while a value closer to -1 indicates a strong negative relationship. Values around 0 indicate no correlation.
""")

# Plot the heatmap
fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='Viridis',
    zmin=-1, zmax=1
))

# Adjust layout for a more square shape
fig.update_layout(
    title='Correlation Heatmap',
    xaxis_nticks=len(corr.columns),
    yaxis_nticks=len(corr.columns),
    width=600,  # Adjust the width as needed
    height=600  # Adjust the height as needed
)

st.plotly_chart(fig, use_container_width=True)

# Big Title
st.title("Model Training")

# Comment about CRISP-DM methodology
st.write("The whole process of exploratory data analysis together with model training and evaluation was made following the CRISP-DM methodology.")

# Introduction to the packages used for model training
st.write("Here follows the packages that were used to train the model:")

# Display the code cell with the imports
st.code("""
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
""", language='python')

# Explanation of what each package does
st.write("""
- **patsy**: A Python library for describing statistical models and building design matrices.
- **statsmodels.api**: Provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests and statistical data exploration.
- **statsmodels.formula.api**: Provides a convenient interface for specifying statistical models using formulas.
- **train_test_split**: A function from scikit-learn that splits the data into random train and test subsets.
- **r2_score**: A function from scikit-learn that computes the coefficient of determination, \(R^2\), to evaluate the goodness of fit of the model.
""")

# Train-test split
st.write("""
### Train-Test Split
The data is split into training and testing sets using an 80-20 split. This allows us to train the model on one portion of the data and test it on another, ensuring that the model generalizes well to unseen data.
""")
st.code("""
train_df, test_df = train_test_split(renda_df, test_size=0.2, random_state=40)
""", language='python')

# Define the formula for the model
st.write("""
### Model Formula
A formula is defined to specify the relationship between the dependent variable (log_renda) and the independent variables.
""")
st.code("""
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
    'idade + '
    'tempo_emprego + '
    'qt_pessoas_residencia + '
    'tempo_emprego_idade_ratio'
)
""", language='python')

st.write("""
Some adjustments were made using the features presented in the formula in order to obtain a better R2 value with the model.
""")

# Create design matrices
st.write("""
### Design Matrices
The Patsy library is used to create design matrices from the formula and the data.
""")
st.code("""
y_train, X_train = patsy.dmatrices(formula_like=formula, data=train_df)
y_test, X_test = patsy.dmatrices(formula_like=formula, data=test_df)
""", language='python')

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

tab = pd.DataFrame({'alpha':alphas,
              'R2 (Ridge)': r2_ridge,  'R2 (Lasso)': r2_lasso})

modelo = sm.OLS(y_train, X_train)
reg = modelo.fit_regularized(method='elastic_net',
                                     refit=True,
                                     L1_wt= 0, #ridge
                                     alpha = 0)

y_pred = reg.predict(X_test)
y_test = y_test.ravel()

predictions = pd.DataFrame({
    'True Value (log)': y_test,
    'Predicted Value (log)': y_pred,
    'Difference (Log)': (y_test - y_pred),
    'True Value': np.exp(y_test), 
    'Predicted Value': np.exp(y_pred),
    'Difference': ((np.exp(y_test)) - (np.exp(y_pred))),
})

st.write("""
### Ridge and Lasso Regression
Ridge and Lasso regression models are fitted using different alpha values. Ridge regression includes a penalty for the sum of squared coefficients (L2 penalty), while Lasso regression includes a penalty for the absolute value of the coefficients (L1 penalty).
""")
st.code("""
alphas = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
r2_ridge = []
r2_lasso = []

for alpha in alphas:
    modelo = sm.OLS(y_train, X_train)
    
    reg_ridge = modelo.fit_regularized(method='elastic_net',
                                     refit=True,
                                     L1_wt=0, # Ridge
                                     alpha=alpha)
    
    reg_lasso = modelo.fit_regularized(method='elastic_net',
                                     refit=True,
                                     L1_wt=1, # Lasso
                                     alpha=alpha)
    
    y_pred_ridge = reg_ridge.predict(X_test)
    y_pred_lasso = reg_lasso.predict(X_test)
    aux = r2_score(y_test, y_pred_ridge)
    tmp = r2_score(y_test, y_pred_lasso)
    
    r2_ridge.append(aux)
    r2_lasso.append(tmp)
""", language='python')

st.write("""
We can use a simple table to capture the relationship between the alpha values and the methods used to train the model.
""")

st.write(tab)

st.write("""
The highest values for R2 were obtained at alpha = 0
""")

# Fit the final model
st.write("""
### Fitting the Final Model
A final Ridge regression model is fitted with alpha set to 0 (equivalent to ordinary least squares regression).
""")
st.code("""
modelo = sm.OLS(y_train, X_train)
reg = modelo.fit_regularized(method='elastic_net',
                                     refit=True,
                                     L1_wt=0, # Ridge
                                     alpha=0)
""", language='python')

# Predict and create a DataFrame for predictions
st.write("""
### Predictions and Evaluation
The final model is used to predict on the test set. A DataFrame is created to compare the true and predicted values, both in log scale and original scale.
""")
st.code("""
y_pred = reg.predict(X_test)
y_test = y_test.ravel()

predictions = pd.DataFrame({
    'True Value (log)': y_test,
    'Predicted Value (log)': y_pred,
    'Difference (Log)': (y_test - y_pred),
    'True Value': np.exp(y_test), 
    'Predicted Value': np.exp(y_pred),
    'Difference': ((np.exp(y_test)) - (np.exp(y_pred))),
})
""", language='python')

# Display the predictions DataFrame
st.write("### Predictions")
st.write(predictions)

# Explanation of the graph
st.write("""
### True vs Predicted Values
The following scatter plot shows the true values versus the predicted values. The red dashed line represents perfect predictions, where the predicted values exactly match the true values.
""")

# Switch button to alternate between log scale and original scale
scale_option = st.radio('Choose scale for the scatter plot:', ('Log Scale', 'Original Scale'))

# Create the scatter plot with Plotly
if scale_option == 'Log Scale':
    fig = px.scatter(predictions, x='True Value (log)', y='Predicted Value (log)', 
                     labels={'True Value (log)': 'True Value (log)', 'Predicted Value (log)': 'Predicted Value (log)'},
                     title='True Values vs Predicted Values (Log Scale)')
    # Add a line for perfect predictions
    max_val = max(predictions[['True Value (log)', 'Predicted Value (log)']].max())
    min_val = min(predictions[['True Value (log)', 'Predicted Value (log)']].min())
    fig.add_shape(type="line",
                  x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color="Red", dash="dash"))
else:
    fig = px.scatter(predictions, x='True Value', y='Predicted Value', 
                     labels={'True Value': 'True Value', 'Predicted Value': 'Predicted Value'},
                     title='True Values vs Predicted Values (Original Scale)')
    # Add a line for perfect predictions
    max_val = max(predictions[['True Value', 'Predicted Value']].max())
    min_val = min(predictions[['True Value', 'Predicted Value']].min())
    fig.add_shape(type="line",
                  x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color="Red", dash="dash"))

# Customize scatter point size
fig.update_traces(marker=dict(size=5))

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

