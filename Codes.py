import datetime as dt
from statistics import stdev
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns

'''
1) Ülkemiz Türkiye'yi baz alacak olursak; 2015-2019 yılları arasında insanların mutlu olmasına etki eden faktörler
yıllara göre neler ve zamanla nasıl değişmiş?

2) Sadece 2019 verisini baz alırsak; 2019 yılındaki en mutlu ülke hangisidir? Hangi faktör/faktörler en mutlu ülke
olmasını sağlamış olabilir? Görselleştirerek ifade etmenizi bekliyoruz.
'''

df_2015 = pd.read_csv("datasets-894-813759-2015.csv")
df_2016 = pd.read_csv("datasets-894-813759-2016.csv")
df_2017 = pd.read_csv("datasets-894-813759-2017.csv")
df_2018 = pd.read_csv("datasets-894-813759-2018.csv")
df_2019 = pd.read_csv("datasets-894-813759-2019.csv")


def wrangle_all_turkey_data():
    df_2015_tr = df_2015[df_2015['Country'] == 'Turkey']
    df_2015_tr.drop(columns=['Region', 'Standard Error'], inplace=True)
    df_2015_tr.rename(columns={'Happiness Rank': 'Overall rank',
                               'Economy (GDP per Capita)': 'GDP per capita',
                               'Happiness Score': 'Score',
                               'Trust (Government Corruption)': 'Perceptions of corruption',
                               'Freedom': 'Freedom to make life choices',
                               'Health (Life Expectancy)': 'Healthy life expectancy'},
                      inplace=True)
    df_2015_tr['Year'] = dt.datetime(2015, 1, 1)

    df_2016_tr = df_2016[df_2016['Country'] == 'Turkey']
    df_2016_tr.drop(columns=['Region'], inplace=True)
    df_2016_tr.rename(columns={'Happiness Rank': 'Overall rank',
                               'Economy (GDP per Capita)': 'GDP per capita',
                               'Happiness Score': 'Score',
                               'Trust (Government Corruption)': 'Perceptions of corruption',
                               'Freedom': 'Freedom to make life choices',
                               'Health (Life Expectancy)': 'Healthy life expectancy'},
                      inplace=True)
    df_2016_tr['Year'] = dt.datetime(2016, 1, 1)

    df_2017_tr = df_2017[df_2017['Country'] == 'Turkey']
    df_2017_tr.drop(columns=['Whisker.high', 'Whisker.low'], inplace=True)
    df_2017_tr.rename(columns={'Happiness.Rank': 'Overall rank',
                               'Economy..GDP.per.Capita.': 'GDP per capita',
                               'Happiness.Score': 'Score',
                               'Trust..Government.Corruption.': 'Perceptions of corruption',
                               'Freedom': 'Freedom to make life choices',
                               'Health..Life.Expectancy.': 'Healthy life expectancy'},
                      inplace=True)
    df_2017_tr['Year'] = dt.datetime(2017, 1, 1)

    df_2018.rename(columns={'Country or region': 'Country'}, inplace=True)
    df_2018_tr = df_2018[df_2018['Country'] == 'Turkey']
    df_2018_tr['Year'] = dt.datetime(2018, 1, 1)

    df_2019.rename(columns={'Country or region': 'Country'}, inplace=True)
    df_2019_tr = df_2019[df_2019['Country'] == 'Turkey']
    df_2019_tr['Year'] = dt.datetime(2019, 1, 1)

    frames = [df_2015_tr, df_2016_tr, df_2017_tr, df_2018_tr, df_2019_tr]
    df = pd.concat(frames)
    return df


def create_full_happiness_factors_2019():
    population_GDP_stats = {'mean': df_2019['GDP per capita'].mean(), 'stddev': stdev(df_2019['GDP per capita'])}
    population_generosity_stats = {'mean': df_2019['Generosity'].mean(), 'stddev': stdev(df_2019['Generosity'])}
    population_health_stats = {'mean': df_2019['Healthy life expectancy'].mean(), 'stddev': stdev(df_2019['Healthy life expectancy'])}
    population_ss_stats = {'mean': df_2019['Social support'].mean(), 'stddev': stdev(df_2019['Social support'])}
    population_life_choices_stats = {'mean':df_2019['Freedom to make life choices'].mean(), 'stddev': stdev(df_2019['Freedom to make life choices'])}
    population_corr_stats = {'mean': df_2019['Perceptions of corruption'].mean(), 'stddev': stdev(df_2019['Perceptions of corruption'])}

    quartile_bands = int(np.floor(df_2019.shape[0]/3))
    quartile1 = df_2019.iloc[:quartile_bands]
    quartile2 = df_2019.iloc[quartile_bands:quartile_bands*2]
    quartile3 = df_2019.iloc[quartile_bands*2:]

    quartile1_gdp_stats = {'mean': quartile1['GDP per capita'].mean(), 'stddev': stdev(quartile1['GDP per capita'])}
    quartile2_gdp_stats = {'mean': quartile2['GDP per capita'].mean(), 'stddev': stdev(quartile2['GDP per capita'])}
    quartile3_gdp_stats = {'mean': quartile3['GDP per capita'].mean(), 'stddev': stdev(quartile3['GDP per capita'])}
    quartile1_ss_stats = {'mean': quartile1['Social support'].mean(), 'stddev': stdev(quartile1['Social support'])}
    quartile2_ss_stats = {'mean': quartile2['Social support'].mean(), 'stddev': stdev(quartile2['Social support'])}
    quartile3_ss_stats = {'mean': quartile3['Social support'].mean(), 'stddev': stdev(quartile3['Social support'])}
    quartile1_lifechoices_stats = {'mean': quartile1['Freedom to make life choices'].mean(), 'stddev': stdev(quartile1['Freedom to make life choices'])}
    quartile2_lifechoices_stats = {'mean': quartile2['Freedom to make life choices'].mean(), 'stddev': stdev(quartile2['Freedom to make life choices'])}
    quartile3_lifechoices_stats = {'mean': quartile3['Freedom to make life choices'].mean(), 'stddev': stdev(quartile3['Freedom to make life choices'])}
    quartile1_generosity_stats = {'mean': quartile1['Generosity'].mean(), 'stddev': stdev(quartile1['Generosity'])}
    quartile2_generosity_stats = {'mean': quartile2['Generosity'].mean(), 'stddev': stdev(quartile2['Generosity'])}
    quartile3_generosity_stats = {'mean': quartile3['Generosity'].mean(), 'stddev': stdev(quartile3['Generosity'])}
    quartile1_health_stats = {'mean': quartile1['Healthy life expectancy'].mean(), 'stddev': stdev(quartile1['Healthy life expectancy'])}
    quartile2_health_stats = {'mean': quartile2['Healthy life expectancy'].mean(), 'stddev': stdev(quartile2['Healthy life expectancy'])}
    quartile3_health_stats = {'mean': quartile3['Healthy life expectancy'].mean(), 'stddev': stdev(quartile3['Healthy life expectancy'])}
    quartile1_corr_stats = {'mean': quartile1['Perceptions of corruption'].mean(), 'stddev': stdev(quartile1['Perceptions of corruption'])}
    quartile2_corr_stats = {'mean': quartile2['Perceptions of corruption'].mean(), 'stddev': stdev(quartile2['Perceptions of corruption'])}
    quartile3_corr_stats = {'mean': quartile3['Perceptions of corruption'].mean(), 'stddev': stdev(quartile3['Perceptions of corruption'])}

    fig = go.Figure()
    fig.add_trace(go.Bar(x=['quartile1', 'quartile2', 'quartile3', 'population'], y=[quartile1_gdp_stats['mean'],
                                                                                     quartile2_gdp_stats['mean'],
                                                                                     quartile3_gdp_stats['mean'],
                                                                                     population_GDP_stats['mean'],
                                                                                     ], name='GDP stats'))

    fig.add_trace(go.Bar(x=['quartile1', 'quartile2', 'quartile3', 'population'], y=[quartile1_ss_stats['mean'],
                                                                                     quartile2_ss_stats['mean'],
                                                                                     quartile3_ss_stats['mean'],
                                                                                     population_ss_stats['mean'],
                                                                                     ], name='Social support stats'))

    fig.add_trace(go.Bar(x=['quartile1', 'quartile2', 'quartile3', 'population'], y=[quartile1_lifechoices_stats['mean'],
                                                                                     quartile2_lifechoices_stats['mean'],
                                                                                     quartile3_lifechoices_stats['mean'],
                                                                                     population_life_choices_stats['mean'],
                                                                                     ], name='Freedom to make life choices stats'))

    fig.add_trace(go.Bar(x=['quartile1', 'quartile2', 'quartile3', 'population'], y=[quartile1_generosity_stats['mean'],
                                                                                     quartile2_generosity_stats['mean'],
                                                                                     quartile3_generosity_stats['mean'],
                                                                                     population_generosity_stats['mean'],
                                                                                     ], name='Generosity'))

    fig.add_trace(go.Bar(x=['quartile1', 'quartile2', 'quartile3', 'population'], y=[quartile1_corr_stats['mean'],
                                                                                     quartile2_corr_stats['mean'],
                                                                                     quartile3_corr_stats['mean'],
                                                                                     population_corr_stats['mean'],
                                                                                     ], name='Perceptions of corruption'))

    fig.add_trace(go.Bar(x=['quartile1', 'quartile2', 'quartile3', 'population'], y=[quartile1_health_stats['mean'],
                                                                                     quartile2_health_stats['mean'],
                                                                                     quartile3_health_stats['mean'],
                                                                                     population_health_stats['mean'],
                                                                                     ], name='Healthy life expectancy'))
    return fig


def create_turkey_happiness_factors_plot(turkey_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=turkey_df.Year, y=turkey_df['GDP per capita'], name='GDP per capita'))
    fig.add_trace(go.Scatter(x=turkey_df.Year, y=turkey_df['Generosity'], name='Generosity'))
    fig.add_trace(go.Scatter(x=turkey_df.Year, y=turkey_df['Healthy life expectancy'], name='Life expectancy'))
    fig.add_trace(go.Scatter(x=turkey_df.Year, y=turkey_df['Perceptions of corruption'], name='Happiness Score'))
    fig.add_trace(go.Scatter(x=turkey_df.Year, y=turkey_df['Freedom to make life choices'], name='Freedom to make life choices'))
    fig.update_layout({'xaxis_title': 'Year',
                       'yaxis_title': 'Index Value',
                       'title': 'Change in Turkish happiness factors between 2015-2019'})
    return fig


def create_turkey_happiness_scores_plot(turkey_df):
    fig_happiness_overall = go.Figure()
    fig_happiness_overall.add_trace(go.Scatter(x=turkey_df.Year, y=turkey_df['Overall rank'], name='Happiness Rank'))
    fig_happiness_overall.add_trace(go.Scatter(x=turkey_df.Year, y=turkey_df['Score'], name='Happiness Score'))
    fig_happiness_overall.show()
    return fig_happiness_overall


def get_correlations(df: pd.DataFrame):
    correlation = df.corr()
    # method 1
    seaborn_ax = sns.heatmap(correlation,
                             xticklabels=correlation.columns.values, yticklabels=correlation.columns.values)

    # method 2
    fig = go.Figure()
    fig.add_heatmap(x=df.keys(), y=df.keys(), z=correlation)

    # method 3
    pairwise_correlations = sns.pairplot(df)

    return seaborn_ax, fig, pairwise_correlations


def main():
    turkey_df = wrangle_all_turkey_data()
    ax_tr, fig_tr, pairwise_tr = get_correlations(turkey_df)
    ax_2019, fig_2019, pairwise_2019 = get_correlations(df_2019)

    fig1 = create_turkey_happiness_factors_plot(turkey_df)
    fig2 = create_turkey_happiness_scores_plot(turkey_df)
    fig3 = create_full_happiness_factors_2019()

    fig1.show()
    fig2.show()
    fig3.show()


if __name__ == '__main__':
    main()

