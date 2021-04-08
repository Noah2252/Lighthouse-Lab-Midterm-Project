import sqlite3

import pandas as pd


def add_weather(df):
    weather = read_weather()
    weather_local = localize_times(weather)
    weather_dummy = add_weather_dummies(weather_local)
    df_with_weather = join_weather_to_df(df, weather_dummy)
    return df_with_weather


def read_weather():
    weather = pd.read_csv('data/weather_events.csv', parse_dates=['start_time_utc', 'end_time_utc'])
    # The times come in as UTC, so tell Pandas that
    weather.start_time_utc = weather.start_time_utc.dt.tz_localize('UTC')
    weather.end_time_utc = weather.end_time_utc.dt.tz_localize('UTC')
    return weather


def localize_times(weather):
    weather_local = weather.copy()
    weather_local['start_time_local'] = weather_local.groupby(
        'time_zone'
    ).start_time_utc.transform(
        lambda x: x.dt.tz_convert(x.name).dt.tz_localize(None)
    )
    weather_local['end_time_local'] = weather_local.groupby(
        'time_zone'
    ).end_time_utc.transform(
        lambda x: x.dt.tz_convert(x.name).dt.tz_localize(None)
    )
    return weather_local


def add_weather_dummies(weather):
    return weather.drop(
        ['event_type', 'severity'], axis=1
    ).join(
        pd.get_dummies(weather.event_type).rename(str.lower, axis=1)
    ).join(
        weather.severity.replace({
            "Light": 1,
            "UNK": 1,
            "Moderate": 2,
            "Heavy": 3,
            "Severe": 3,
            "Other": 3,
        })
    )


query = """
select flights.id,
max(wo.cold) as cold_o, max(wo.fog) as fog_o, max(wo.hail) as hail_o,
max(wo.precipitation) as precipitation_o,
max(wo.rain) as rain_o, max(wo.snow) as snow_o, max(wo.storm) as storm_o,
max(wo.severity) as severity_o,
max(wd.cold) as cold_d, max(wd.fog) as fog_d, max(wd.hail) as hail_d,
max(wd.precipitation) as precipitation_d,
max(wd.rain) as rain_d, max(wd.snow) as snow_d, max(wd.storm) as storm_d,
max(wd.severity) as severity_d
from flights
left join weather as wo
on (
    flights.origin = wo.iata_code
    and flights.fl_date_crs_dep_time between
    wo.start_time_local and wo.end_time_local
)
left join weather as wd
on (
    flights.dest = wd.iata_code
    and flights.fl_date_crs_arr_time between
    wd.start_time_local and wd.end_time_local
)
group by flights.id;
"""


def join_weather_to_df(df, weather):
    df_sel = df[[
        'origin', 'fl_date_crs_dep_time',
        'dest', 'fl_date_crs_arr_time',
    ]]
    weather_sel = weather[[
        'iata_code', 'start_time_local', 'end_time_local',
        'cold', 'fog', 'hail', 'precipitation', 'rain', 'snow', 'storm',
        'severity',
    ]]
    
    with sqlite3.connect(':memory:') as conn:
        df_sel.to_sql('flights', conn)
        weather_sel.to_sql('weather', conn, index=False)
        conn.cursor().execute(
            'create index weather_idx on weather (iata_code, start_time_local, end_time_local)'
        )
        
        result = pd.read_sql_query(query, conn).fillna(0)
        for col in result.columns:
            result[col] = result[col].astype(int)
        
        return df.join(result.set_index('id'))
