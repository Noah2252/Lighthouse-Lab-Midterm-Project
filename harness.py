import pandas as pd
import joblib
import sklearn.metrics as met


y_col = 'arr_delay'
submission_columns = [
    'fl_date',
    'mkt_carrier',
    'mkt_carrier_fl_num',
    'origin',
    'dest'
]


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
    
    return clean_test(x_no_na)


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

def dummy_maker(df,col):
        """
        A function that takes a Dataframe and a catagorical column name and returns dataframe with the column replaced by dummy variables.
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
    Takes a DataFrame and a catagorical column name and adds five new columns to the Dataframe base off of grouped delay stats in relation to the catagories.
    Parameters:
        - df: The dataframe.
        - col (str): The catagoriclal column which you would like to produces stats from in relation to delay.
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
        return met.r2_score(y, y_pred)
    
    def submit(self, x_path, submission_path, y_column_name):
        """
        Creates the submission file by predicting (untransformed) y
        values for the (untransformed) x values at x_path.
        """
        x = pd.read_csv(x_path)
        x_tr = self.transformers.x_transformer(x)
        y_tr = self.model.predict(x_tr)
        y_tr = pd.DataFrame(y_tr, index=x.index, columns=[y_column_name])
        y = self.transformers.y_untransformer(y_tr)
        x = x[submission_columns]
        x[y_column_name] = y
        x.to_csv(submission_path, index=False)
        
