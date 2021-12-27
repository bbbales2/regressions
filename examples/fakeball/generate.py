from faker import Faker
import itertools
import math
import numpy
import pandas
import random
import scipy
import scipy.stats
import datetime

N_teams = 10
N_players = 10
N_rounds = 2
N_shots = 80
N_attempts = 1
N_games_per_day = 5

fake = Faker()

rows = []
for t in range(N_teams):
    for p in range(N_players):
        team = f"team{t:02d}"
        player = fake.unique.name()
        offense = scipy.stats.norm.rvs()
        defense = scipy.stats.norm.rvs()

        rows.append({"team": team, "player": player, "offense": offense, "defense": defense})

players_df = pandas.DataFrame.from_records(rows)

teams = set(players_df["team"])

one_day = datetime.timedelta(days=1)
date = datetime.datetime.strptime("2021-10-01", "%Y-%m-%d")

rows = []
game_count = 0
for round in range(N_rounds):
    team_pairs = list(itertools.combinations(teams, 2))
    random.shuffle(team_pairs)
    for team1, team2 in team_pairs:
        print(f"Playing {team1} vs. {team2}")
        team1_df = players_df[players_df.team == team1]
        team2_df = players_df[players_df.team == team2]
        for shot in range(N_shots):
            team1_active_df = team1_df.sample(n=5)
            team2_active_df = team2_df.sample(n=5)
            logit_mu = team1_active_df["offense"].sum() - team2_active_df["defense"].sum()

            offense_players = {f"o{i}": name for i, name in enumerate(team1_active_df["player"])}

            defense_players = {f"d{i}": name for i, name in enumerate(team2_active_df["player"])}

            for attempt in range(N_attempts):
                made = scipy.stats.bernoulli.rvs(scipy.special.expit(logit_mu))

                row = {
                    "team1": team1,
                    "team2": team2,
                    "made": made,
                    "date": (date + math.floor(game_count / N_games_per_day) * one_day).strftime("%Y-%m-%d"),
                    **offense_players,
                    **defense_players,
                }

                rows.append(row)
        game_count += 1

shots_df = pandas.DataFrame.from_records(rows)

players_df.to_csv("examples/fakeball/players.csv", index=False)
shots_df.to_csv("examples/fakeball/shots.csv", index=False)
