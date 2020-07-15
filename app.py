import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import math

fontname = 'Enigmatic Unicode'

def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

def get_plot_layout(n):
    if n % 2 == 1:
        n += 1
    if (n == 16) or (n == 26):
        n += 2
    divisors = np.array(list(divisorGenerator(n))).astype('int')
    return divisors[int(len(divisors) / 2 - 1): -int(len(divisors) / 2 - 1)]

def add_season(d):
    d['Year'] = d.date.astype('str').apply(lambda x: x.split('-')[0])
    d['Month'] = d.date.astype('str').apply(lambda x: x.split('-')[1])
    d['Day'] = d.date.astype('str').apply(lambda x: x.split('-')[2])
    d[['Year', 'Month', 'Day']] = d[['Year', 'Month', 'Day']].astype('int')
    d['date'] = pd.to_datetime(d['date'])
    d['day_name'] = d['date'].dt.day_name()
    d['month_name'] = d['date'].dt.month_name()

    d.loc[((d.Year == 2016)&(d.Month >= 7)|(d.Year == 2017)&(d.Month < 7)),
                  'Season'] = 2016
    d.loc[((d.Year == 2017)&(d.Month >= 7)|(d.Year == 2018)&(d.Month < 7)),
                  'Season'] = 2017
    d.loc[((d.Year == 2018)&(d.Month >= 7)|(d.Year == 2019)&(d.Month < 7)),
                  'Season'] = 2018
    d.loc[((d.Year == 2019)&(d.Month >= 7)|(d.Year == 2020)&(d.Month < 8)),
                  'Season'] = 2019
    d['Season'] = d['Season'].astype('float')
    return d

def get_outstanding_games(team):
    return d.loc[pd.isna(d.score1)
                 &((d.team1 == team)|((d.team2 == team)))]

def get_next_opponent(team):
    outstanding_games = get_outstanding_games(team)
    next_opponent = outstanding_games[['team1', 'team2']].iloc[0].tolist()
    next_opponent.remove(team)
    return next_opponent[0]

def get_last_n_games(d, team, n = 5, season = None):
    games = d.loc[pd.notna(d.score1)
                 &((d.team1 == team)|((d.team2 == team)))]
    if season is not None:
        available_seasons = games.Season.unique().tolist()
        if season in available_seasons:
            games = games.loc[games.Season == season]
        else:
            seasons_string = ', '.join([str(year) for year in available_seasons])
            raise ValueError(f'Season {season} is not a valid season for {team}. {seasons_string} are valid seasons.')
    
    games = games.iloc[-n:]
    games[['score1', 'score2']] = games[['score1', 'score2']].astype('int')
    return games

def add_value_by_team(games, team, column):
    values_for = []
    values_against = []

    for idx, row in games.iterrows():
        values_for.append(row[column+'1'] if row['team1'] == team else row[column+'2'])

    for idx, row in games.iterrows():
        values_against.append(row[column+'1'] if row['team1'] != team else row[column+'2'])

    games[column+'_for'] = values_for
    games[column+'_against'] = values_against
    return games

def add_all_values_by_team(games, team):
    games = add_value_by_team(games, team, 'xg')
    games = add_value_by_team(games, team, 'nsxg')
    games = add_value_by_team(games, team, 'score')
    return games

def add_opponents(games, team):
    games['Opponent'] = games.apply(lambda x: x.team1 if x.team1 != team else x.team2, axis = 1).values
    return games

def add_points_won(games):
    games['points'] = np.where(games['score_for'] > games['score_against'], 3,
                               np.where(games['score_for'] == games['score_against'], 1, 0))
    return games

def add_ground(games, team):
    games['ground'] = games.team1.apply(lambda x: 'Home' if x == team else 'Away')
    return games

def get_games(d, team, n = 5, season = 2019):
    games = get_last_n_games(d, team, n = n, season = season)
    games = add_all_values_by_team(games, team)
    games = add_opponents(games, team)
    games = add_points_won(games)
    games = add_ground(games, team)

    return games


@st.cache(suppress_st_warning=True)
def get_data():
    d = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv')
    d = add_season(d)

    # available leagues with xg data
    leagues = d.dropna().league.unique()
    seasons = d.dropna().Season.unique().astype('int')[-1::-1]

    return d, leagues, seasons

df, leagues, seasons = get_data()

league_option = st.sidebar.selectbox(
    "Which League do you want to view?",
    leagues)

season_option = st.sidebar.selectbox(
    "Which Season do you want to view?",
    seasons)

d = df.loc[(df.league == league_option)&(df.Season == season_option)&(pd.notna(df['xg1']))]
club_names = d.team1.unique()

games_list = {}
xgs_list = {}

for club in club_names:
    games = get_games(d, club, n = 0, season=season_option)
    games_list[club] = games
    xgs_list[club] = games[['xg_for', 'xg_against']].rolling(window = 7, min_periods = 1).mean()


n_teams = len(club_names)
width, height = get_plot_layout(n_teams)
fig, ax = plt.subplots(height, width, figsize = (height*5, width*8), sharex=False, sharey=False)
ax = ax.ravel()

for idx, club in enumerate(sorted(club_names)):
    plt.sca(ax[idx])

    for key, df in xgs_list.items():
        plt.plot(np.arange(1, len(df)+1), (df.xg_for - df.xg_against), color='k', alpha = 0.15)

        if key == club:
            plt.plot(np.arange(1, len(df)+1), (df.xg_for - df.xg_against), color='firebrick', alpha = 1)

    plt.xticks(np.arange(4, len(df)+1, 4))
    plt.ylim(-2.5, 2.5)
    
    ymin, ymax = plt.gca().get_ylim()
    plt.hlines(y=0, xmin=0., xmax=plt.gca().get_xlim()[1], color='k', lw=0.5)

    sns.despine()
    plt.title(club, color='firebrick', fontsize=20, fontname=fontname)
    
    if idx >= (n_teams - width):
        plt.xlabel('Gameweek', fontname=fontname, fontsize=16)
    if idx % width == 0:
        plt.ylabel('Expected Goals Difference', fontname=fontname, fontsize=16)
    
plt.suptitle(f'Rolling Expected Goal Difference (over 7 games)\n{league_option} - {season_option}/{season_option+1}', fontsize=42, fontname=fontname)


fig.text(s = 'Data: FiveThirtyEight', x=0.01, y=0.0025, fontsize=14, fontname = fontname, color = '#969696')
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.275)

st.pyplot()