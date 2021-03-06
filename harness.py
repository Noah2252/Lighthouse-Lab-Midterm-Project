import pandas as pd
import numpy as np
import joblib
import sklearn.preprocessing as pre
import sklearn.metrics as met

import weather

y_col = 'arr_delay'
submission_columns = [
    'fl_date',
    'mkt_carrier',
    'mkt_carrier_fl_num',
    'origin',
    'dest'
]


def add_colon(time_str):
    return (time_str[:-2] + ":" + time_str[-2:]).zfill(4)


def remove_colon(time_col):
    return time_col.str.replace(':', '').astype(int)


def read_flights(flights_path):
    result = pd.read_csv(
        flights_path,
        index_col=0,
        converters={
            'fl_date': lambda x: x[:10],
            'crs_dep_time': add_colon,
            'crs_arr_time': add_colon,
        },
        parse_dates=[
            ['fl_date', 'crs_dep_time'],
            ['fl_date', 'crs_arr_time']
        ],
        keep_date_col=True,
    ).reset_index()
    if 'id' in result.columns:
        result.set_index('id', inplace=True)
    result.crs_dep_time = remove_colon(result.crs_dep_time)
    result.crs_arr_time = remove_colon(result.crs_arr_time)
    return result


def clean_train(x):
    """
    Cleans the training data (e.g. removing nulls and outliers).
    """
    x_no_na = x.dropna(
        subset=[
            'tail_num', 'dep_time', 'dep_delay', 'taxi_out',
            'wheels_off', 'wheels_on', 'taxi_in', 'arr_time',
            'arr_delay', 'actual_elapsed_time', 'air_time'
        ]
    )
    x_no_na = x_no_na.drop(['cancellation_code','carrier_delay',
                    'weather_delay','nas_delay','security_delay',
                    'late_aircraft_delay','first_dep_time',
                   'total_add_gtime','longest_add_gtime',
                   'no_name'], axis =1)
    return clean_test(x_no_na)


def clean_train_late_only(x):
    x = clean_train(x)
    return x[x.arr_delay > 0]
    

def clean_test(x):
    """
    Cleans the testing data. This can't remove
    rows, but can do other cleaning like
    standardizing representations and imputing
    missing values. 
    """
    return x


def save(model, path):
    """
    Saves the specified trained model to the specified
    file path.
    """
    joblib.dump(model, path + '.joblib')


def load(path):
    """
    Loads the trained model from the specified file path.
    """
    return joblib.load(path + '.joblib')


def normalize(y):
    normalizer = load('y_normalizer')
    y_tr = normalizer.transform(y)
    y_tr = pd.DataFrame(y_tr, index=y.index, columns=y.columns)
    return y_tr


def unnormalize(y_tr):
    normalizer = load('y_normalizer')
    y = normalizer.inverse_transform(y_tr)
    y = pd.DataFrame(y, index=y_tr.index, columns=y_tr.columns)
    return y


def keep_only(df, *cols):
    return df[list(cols)]


def drop(df, *cols):
    return df.drop(list(cols), axis=1)


def add_hour_decimal(df, col):
    result = df.copy()
    result[col + '_hours'] = result[col] // 100 + (result[col] % 100) / 60
    return result


def add_all_hour_decimal(df):
    df = add_hour_decimal(df, 'crs_dep_time')
    df = add_hour_decimal(df, 'crs_arr_time')
    return df


def powers_of(df, degree):
    column_name = df.columns[0]
    scaled = pre.StandardScaler().fit_transform(df)
    dep_powers = pre.PolynomialFeatures(degree=degree).fit_transform(scaled)[:, 1:]
    return pd.DataFrame(
        dep_powers, columns = [
            f'{column_name}_{i + 1}' for i in range(degree)
        ],
        index=df.index
    )


def powers_of_time(df):
    df = add_all_hour_decimal(df)
    return df.join(powers_of(df[['crs_dep_time_hours']], 6)).join(
        powers_of(df[['crs_arr_time_hours']], 6)
    )


def dummy_maker(df,col):
    """
    A function that takes a Dataframe and a catagorical column name and returns
    dataframe with the column replaced by dummy variables.
    Parameters
        - df: The Dataframe
        - col (str): the column name
    Returns:
        - df_2: The Dataframe with dummbies instead of the selected columns
    """
    dummy = pd.get_dummies(df[col])
    df_2 = pd.concat([df,dummy], axis=1)
    df_2 = df_2.drop(col,axis=1)
    return df_2


def add_grouped_stats(df,col):
    """
    Takes a DataFrame and a catagorical column name and adds five new columns to the
    Dataframe base off of grouped delay stats in relation to the catagories.
    Parameters:
        - df: The dataframe.
        - col (str): The catagoriclal column which you would like to produces
          stats from in relation to delay.
    Returns:
        - df_2: The new dataframe with the stat column based off of chosen col.
    """
    df_2 = df.copy()
    col_mean=col + '_delay_mean'
    col_median=col+'_delay_median'
    col_std=col+'_delay_std'
    col_min = col+'_delay_min'
    col_max=col+'_delay_max'
    df_2[col_mean] = df[col].map(df.groupby([col]).arr_delay.mean().to_dict())
    df_2[col_median] = df[col].map(df.groupby([col]).arr_delay.median().to_dict())
    df_2[col_std] = df[col].map(df.groupby([col]).arr_delay.std().to_dict())
    df_2[col_min] = df[col].map(df.groupby([col]).arr_delay.min().to_dict())
    df_2[col_max] = df[col].map(df.groupby([col]).arr_delay.max().to_dict())
    return df_2


def transfer_grouped_stats(df_train, df_test, col):
    """
    Takes a DataFrame and a categorical column name and adds grouped
    stats for that column from another DataFrame. Use this
    to transfer the training data's grouped stats to the testing data.
    
    Categories in the test data that aren't found in the training
    data get the overall mean and median (across all categories)
    the biggest std and max from any category, and the smallest min
    from any category.
    Parameters:
        - df_train: The dataframe with the training data, from which
          group stats should be calculated.
        - df_test: The dataframe with the testing data, which the
          new columns should be added to.
        - col (str): The categorical column which you would like to produces
          stats from in relation to delay.
    Returns:
        - df_2: The new dataframe with the stat column based off of chosen col.
    """
    df_2 = df_test.copy()
    col_mean=col + '_delay_mean'
    col_median=col+'_delay_median'
    col_std=col+'_delay_std'
    col_min = col+'_delay_min'
    col_max=col+'_delay_max'
    
    df_2[col_mean] = df_test[col].map(
        df_train.groupby([col]).arr_delay.mean().to_dict()
    ).astype(float).fillna(df_train.arr_delay.mean())
    df_2[col_median] = df_test[col].map(
        df_train.groupby([col]).arr_delay.median().to_dict()
    ).astype(float).fillna(df_train.arr_delay.median())
    
    std_map = df_train.groupby([col]).arr_delay.std().to_dict()
    df_2[col_std] = df_test[col].map(std_map).astype(float).fillna(
        max(std_map.values())
    )
    min_map = df_train.groupby([col]).arr_delay.min().to_dict()
    df_2[col_min] = df_test[col].map(min_map).astype(float).fillna(
        min(min_map.values())
    )
    max_map = df_train.groupby([col]).arr_delay.max().to_dict()
    df_2[col_max] = df_test[col].map(max_map).astype(float).fillna(
        max(max_map.values())
    )
    
    return df_2


def transfer_grouped_means(df, df_train, max_groups, *cols):
    col_mean = "__".join(cols) + "_mean"
    if max_groups is None:
        means = df_train.groupby(list(cols)).arr_delay.mean().rename(col_mean)
    else:
        groups = (
            df_train
            .groupby(['origin', 'dest'])
            .agg({'arr_delay': ['count', 'mean']})
        )
        groups.columns = groups.columns.droplevel()
        means = (
            groups
            .sort_values(by='count', ascending=False)
            .head(max_groups)
            .drop('count', axis=1)
            .rename({'mean': col_mean}, axis=1)
        )
        
    result = df.merge(
        means, left_on=['origin', 'dest'], right_index=True, how='left'
    ).reindex(df.index)
    result['has_' + col_mean] = (~result[col_mean].isna()).astype(int)
    return result.fillna(df_train.arr_delay.mean())


def chain(*funcs):
    """
    Chains several functions together, passing the output of each function
    as the first argument of the next. The result is a function
    that takes the first function's first argument and returns the
    last function's output.
    
    Each argument must be a function or a list containing the
    function and the other arguments to pass to it.
    """
    def chained(arg):
        for func in funcs:
            try:
                f, *args = func
            except TypeError:
                f = func
                args = []
            arg = f(arg, *args)
        return arg
    return chained


def keep_only_test_columns(df):
    return df[[
        'fl_date_crs_dep_time',
        'fl_date_crs_arr_time',
        'fl_date',
        'mkt_unique_carrier',
        'branded_code_share',
        'mkt_carrier',
        'mkt_carrier_fl_num',
        'op_unique_carrier',
        'tail_num',
        'op_carrier_fl_num',
        'origin_airport_id',
        'origin',
        'origin_city_name',
        'dest_airport_id',
        'dest',
        'dest_city_name',
        'crs_dep_time',
        'crs_arr_time',
        'crs_elapsed_time',
        'distance',
    ]]


def make_all_dummies(df, df_train):
    df = make_all_dummies_except_city_pairs(df, df_train)
    df = make_city_pairs_dummies(df, df_train)
    return df


def make_all_dummies_except_city_pairs(df, df_train):
    df = make_weather_dummies(df, df_train)
    df = make_city_dummies(df, df_train)
    df = make_date_dummies(df, df_train)
    df = make_hour_dummies(df, df_train)
    df = make_carrier_dummies(df, df_train)
    df = make_haul_dummies(df, df_train)
    return df


def make_weather_dummies(df, df_train):
    return weather.add_weather(df)

    
def make_city_dummies(df, df_train):
    cols = [
        'origin_city_name', 'dest_city_name',
        'origin_airport_id', 'dest_airport_id',
    ]
    for col in cols:
        df = make_dummies(df, df_train, col)
    return df


def make_date_dummies(df, df_train):
    cols = ['month', 'day']
    for col in cols:
        df = make_dummies(df, df_train, col)
    return df
    

def make_hour_dummies(df, df_train):
    return make_dummies(df, df_train, 'hour')
    

def make_carrier_dummies(df, df_train):
    return make_dummies(df, df_train, 'op_unique_carrier')


def make_haul_dummies(df, df_train):
    return make_dummies(df, df_train, 'haul')


def make_city_pairs_dummies(df, df_train):
    df = df.copy()
    df['origin_dest_city_name'] = (
        df['origin_city_name'] +' to '+ df['dest_city_name']
    )
    df_train = df_train.copy()
    df_train['origin_dest_city_name'] = (
        df_train['origin_city_name'] + ' to ' + df_train['dest_city_name']
    )
    
    # Use only the 500 most common city pairs
    counts = df_train.origin_dest_city_name.value_counts()
    mask = df_train.origin_dest_city_name.isin(counts.head(500).index)
    df_train.origin_dest_city_name = df_train.origin_dest_city_name.where(mask, 'Other')
    
    return make_dummies(df, df_train, 'origin_dest_city_name')


def make_dummies(df, df_train, col):
    """
    Creates dummies for a column in df, using
    categories taken from def_train. This ensures that the
    same columns appear in the same order in both training and
    testing.
    
    Any categories present in df_train but not in df are
    added with all zeros; any categories present in df
    but not in df_train are removed.
    """
    categories = sorted(df_train[col].unique())
    dummy_cols = [f'{col}_{cat}' for cat in categories]
    dummy = pd.get_dummies(df[col], prefix=col)
    for missing_dummy in (set(dummy_cols) - set(dummy.columns)):
        dummy[missing_dummy] = 0
    dummy = dummy[dummy_cols]
    return pd.concat([df, dummy], axis=1)
    

def add_date_parts(df):
    result = df.copy()
    result['month']=result.fl_date.map(lambda v: int(v[5:7]))
    result['day']=result.fl_date.map(lambda v: int(v[8:10]))
    return result


def add_hour(df):
    result = df.copy()
    result['hour']=result.crs_dep_time.map(lambda v: np.floor(v/100))
    return result
    
def add_haul(df):
    result = df.copy()
    result['haul']=result.crs_elapsed_time/60
    result['haul'] = pd.cut(
        result.haul,bins=[-99,3,6,99],labels=[1,2,3]
    ).astype(int)
    return result


def read_weather():
    weather_0 = pd.read_csv('weather_0.csv', index_col=0)
    weather_1 = pd.read_csv('weather_1.csv', index_col=0)
    weather_2 = pd.read_csv('weather_2.csv', index_col=0)
    weather_3 = pd.read_csv('weather_3.csv', index_col=0)
    weather_4 = pd.read_csv('weather_4.csv', index_col=0)
    weather = pd.concat([weather_0,weather_1,weather_2,weather_3,weather_4])
    return weather


def add_weather(df):
    weather = read_weather()
    df_with_weather = df.merge(
        weather, left_on=['fl_date', 'origin_city_name'], right_on=['date', 'city']
    ).merge(
        weather, left_on=['fl_date', 'dest_city_name'], right_on=['date', 'city'],
        suffixes=('_origin', '_dest')
    )

    weather_category_map = {
        'Partially cloudy': 'Cloudy',
        'Clear': 'Sunny',
        'Rain, Partially cloudy': 'Rainy',
        'Rain, Overcast': 'Rainy',
        'Overcast': 'Cloudy',
        'Rain': 'Rainy',
        'Snow, Partially cloudy': 'Snowy',
        'Snow, Overcast': 'Snowy',
        'Snow': 'Snowy',
    }
    df_with_weather['weather_origin'] = df_with_weather.conditions_origin.map(weather_category_map)
    df_with_weather['weather_dest'] = df_with_weather.conditions_dest.map(weather_category_map)
    
    df_with_weather = df.join(
        df_with_weather[
            ['conditions_origin', 'conditions_dest', 'weather_origin', 'weather_dest']
        ],
        how='left'
    )
    
    df_with_weather['conditions_origin'] = df_with_weather['conditions_origin'].fillna('Unknown')
    df_with_weather['conditions_dest'] = df_with_weather['conditions_dest'].fillna('Unknown')
    df_with_weather['weather_origin'] = df_with_weather['weather_origin'].fillna('Unknown')
    df_with_weather['weather_dest'] = df_with_weather['weather_dest'].fillna('Unknown')
    
    return df_with_weather


def add_all_grouped_stats(df, df_train):
    df = add_weather_grouped_stats(df, df_train)
    df = add_city_grouped_stats(df, df_train)
    df = add_date_grouped_stats(df, df_train)
    df = add_carrier_grouped_stats(df, df_train)
    df = add_haul_grouped_stats(df, df_train)
    df = add_tail_num_grouped_stats(df, df_train)
    df = add_hour_grouped_stats(df, df_train)
    return df


def add_weather_grouped_stats(df, df_train):
    cols = [
        'conditions_origin', 'conditions_dest',
        'weather_origin', 'weather_dest',
    ]
    for col in cols:
        df = transfer_grouped_stats(df_train, df, col)
    return df


def add_city_grouped_stats(df, df_train):
    cols = [
        'origin_city_name', 'dest_city_name',
        'origin_airport_id', 'dest_airport_id',
    ]
    for col in cols:
        df = transfer_grouped_stats(df_train, df, col)
    return df


def add_date_grouped_stats(df, df_train):
    cols = [
        'day', 'month',
    ]
    for col in cols:
        df = transfer_grouped_stats(df_train, df, col)
    return df


def add_hour_grouped_stats(df, df_train):
    cols = [
        'hour'
    ]
    for col in cols:
        df = transfer_grouped_stats(df_train, df, col)
    return df


def add_carrier_grouped_stats(df, df_train):
    return transfer_grouped_stats(df_train, df, 'op_unique_carrier')


def add_haul_grouped_stats(df, df_train):
    return transfer_grouped_stats(df_train, df, 'haul')


def add_tail_num_grouped_stats(df, df_train):
    return transfer_grouped_stats(df_train, df, 'tail_num')


def only_numeric(df):
    return df.select_dtypes('number').drop([
        'mkt_carrier_fl_num', 'op_carrier_fl_num',
        'origin_airport_id', 'dest_airport_id'
    ], axis=1, errors='ignore')


def scale(df):
    scaled = pre.StandardScaler().fit_transform(df)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def remove_early(y):
    return y.mask(y < 0, 0)


def discard_early(y_true, y_pred):
    return y_true[y_true > 0].dropna(), y_pred[y_true > 0].dropna()


def score(y_true, y_pred):
    return Score(
        {
            "R squared": met.r2_score(y_true, y_pred),
#             "Median absolute error": met.median_absolute_error(y_true, y_pred),
            "R squared (early = 0)": met.r2_score(
                remove_early(y_true), remove_early(y_pred)
            ),
#             "Median absolute error (no early)": met.median_absolute_error(
#                 remove_early(y_true), remove_early(y_pred)
#             ),
            "R squared (only delay)": met.r2_score(
                *discard_early(y_true, y_pred)
            )
        }
    )


class Score:
    def __init__(self, scores):
        self.scores = scores
    
    def __repr__(self):
        return "\n".join(f"{metric}: {score:.3g}" for metric, score in self.scores.items())


class DataTransformer:
    def __init__(self, x_transformer, y_transformer=None, y_untransformer=None):
        """
        Creates a DataTransformer with the specified transformers.
        
        The x transformer is a function that transforms the independent
        variables dataframe into the form expected by the model.
        This includes throwing away unused columns, creating
        split or combined columns, scaling and normalizing data,
        and converting categorical columns to numeric columns.
        
        The y transformer is a function that transforms the dependent
        variables (i.e. training labels) into the form needed to train
        the model. For example, this might involve normalizing the
        values or applying a Box-Cox transform.
        
        The y untransformer is a function that transforms the model's
        predictions into something comparable against the target
        values. This should be the inverse of the y transformer.
        """
        self.x_transformer = x_transformer
        self.y_transformer = y_transformer or (lambda y: y)
        self.y_untransformer = y_untransformer or (lambda y: y)
    
    @staticmethod
    def extract_x_y(xy):
        """
        Separates a dataframe containing both the independent and
        dependent variables into two dataframes, one for the
        independent variables and one for the dependent variables.
        """
        x = xy.drop(y_col, axis=1)
        y = xy[[y_col]]
        return x, y
    
    def extract_transform(self, xy):
        """
        Separates the dataframe into the independent and dependent
        variables and applies the x and y transformers.
        """
        x, y = self.extract_x_y(xy)
        return self.x_transformer(x), self.y_transformer(y)
    
    def score_transformed(self, estimator, x_tr, y):
        y_pred_tr = estimator.predict(x_tr)
        y_tr = self.y_transformer(pd.DataFrame(y))
        return met.r2_score(y_tr, y_pred_tr)
    
    def score_untransformed(self, estimator, x_tr, y_tr):
        y_pred_tr = estimator.predict(x_tr)
        y_pred = self.y_untransformer(pd.DataFrame(y_pred_tr))
        y = self.y_untransformer(pd.DataFrame(y_tr))
        return met.r2_score(y, y_pred)
    
    
class TrainedModel:
    def __init__(self, model, transformers):
        """
        Creates a TrainedModel with the specified trained sklearn model
        and transformers.
        """
        self.model = model
        self.transformers = transformers
    
    def validate(self, valid_xy):
        """
        Validates the specified model by predicting y values
        from the specified (untransformed) x dataframe. Returns the
        r-squared score for the predictions.
        """
        x, y = self.transformers.extract_x_y(valid_xy)
        x_tr = self.transformers.x_transformer(x)
        y_pred_tr = self.model.predict(x_tr)
        y_pred_tr = pd.DataFrame(y_pred_tr, index=y.index, columns=y.columns)
        y_pred = self.transformers.y_untransformer(y_pred_tr)
        return x_tr, y, y_pred, score(y, y_pred)
    
    def submit(self, x_path, submission_path, y_column_name):
        """
        Creates the submission file by predicting (untransformed) y
        values for the (untransformed) x values at x_path.
        """
        x = clean_test(read_flights(x_path))
        x_tr = self.transformers.x_transformer(x)
        y_tr = self.model.predict(x_tr)
        y_tr = pd.DataFrame(y_tr, index=x.index, columns=[y_column_name])
        y = self.transformers.y_untransformer(y_tr)
        x = x[submission_columns]
        x[y_column_name] = y
        x.to_csv(submission_path, index=False)
        
